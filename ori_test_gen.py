def generate_testset(self):
    # enable this function in core.py to generate points
    goal_range = self.goal_range_high - self.goal_range_low
    rows = int(round((goal_range[0] * 20 + 1) * (goal_range[1] * 20 + 1) * (goal_range[2] * 20 + 1) * 5))
    save_goals = np.zeros((rows, 7))
    counter = 0
    for i in range(int(round((goal_range[0] * 20 + 1)))):
        for j in range(int(round(goal_range[1] * 20 + 1))):
            for k in range(int(round(goal_range[2] * 20 + 1))):
                for w in range(5):
                    goal = save_goals[counter, :]
                    valid = False
                    tries = 0
                    while valid is False:
                        if tries > 0:
                            print("retrying {} times".format(tries))
                        tries += 1
                        goal_pos = np.zeros(3)
                        goal_pos[0] = i / 20 + self.goal_range_low[0]
                        goal_pos[1] = j / 20 + self.goal_range_low[1]
                        goal_pos[2] = k / 20 + self.goal_range_low[2]
                        goal_rot = np.array(Quaternion.random().elements)
                        goal = np.concatenate((goal_pos, np.roll(goal_rot, -1)))
                        angles = ur5e.inverse(goal, False)
                        if angles is None or np.max(np.abs(angles)) > 6.28:
                            pass
                        else:
                            self.robot.set_joint_angles(angles)
                            self.sim.step()
                            valid = self.is_success(goal, self.get_achieved_goal())
                    self.robot.reset()

                    save_goals[counter, :] = goal
                    counter += 1

    np.savetxt("testset_ori.txt", save_goals)