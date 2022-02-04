class PSet4():

    def solve_pddl(self, domain_file: str, problem_file: str):
        """

        domain_file: str - the path to the domain PDDL file
        problem_file: str - the path to the problem PDDL file

        returns: a list of Action as Strings , or None if problem is infeasible
        """

        plan = ['Action 1','Action 2'] 
        return plan

    def solve_rrt(self, corners: [(float, float)]):
        """
        corners: [(float, float)] - a list of 4 (x, y) corners in a rectangle, in the
           order upper-left, upper-right, lower-right, lower-left

        returns: a list of (float_float) tuples containing the (x, y) positions of
           vertices along the path from the start to the goal node. The 0th index
           should be the start node, the last item should be the goal node. If no
           path could be found, return None
        """

        return None


if __name__ == "__main__":
    p = PSet4()
    plan = p.solve_pddl('domain.pddl', 'problem.pddl')
    print("Plan : ", plan)
    print("Plan Length : ", len(plan))
    rrt_path = p.solve_rrt([()])
    print("Path length: " + len(rrt_path))
