import cvxpy as cp
import logging


def _try_solve_problem(problem, w):
    """Try to solve the optimization problem with default and SCS solvers.

    Args:
        problem: The CVXPY optimization problem.
        w: Weights.

    Returns:
        Loss value.
    """
    try:
        loss_value = problem.solve()
    except Exception as ex:
        logging.warning(f"Solver error: {ex}")

    if w.value is None:
        logging.info(f"Can't solve problem with OSQP. Trying another solver...")
        loss_value = problem.solve(solver=cp.SCS)
    return loss_value

