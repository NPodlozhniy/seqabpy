import io
from contextlib import redirect_stdout
from typing import Union

from scipy.optimize import brentq as root
from scipy.stats import multivariate_normal
from statsmodels.sandbox.distributions.extras import mvnormcdf

from seqabpy import *


def alpha_spending_function(
    t: np.ndarray, iuse: int = 1, phi: float = 1, alpha: float = 0.05
) -> tuple:
    """
    This function implements different alpha spending functions commonly used in sequential
    clinical trials or other statistical analyses where early stopping is possible.
    It allows you to choose from several established methods for controlling the overall
    Type I error rate (alpha) as data accumulates.

    Args:
        t: A number or array of numbers representing the proportion of the total
           sample size or information time.
        iuse: An integer specifying the alpha spending function to use:
              1: O'Brien-Fleming
              2: Pocock
              3: Kim-DeMets
              4: Hwang-Shih-DeCani
              5: Haybittle-Peto
        phi: A parameter used in the Kim-DeMets and Hwang-Shih-DeCani spending functions.
        alpha: The overall desired Type I error rate.

    Returns:
        The calculated alpha spending values at the given time points.

    References:
        * O'Brien, P. C., and Fleming, T. R. (1979). A multiple testing
          procedure for clinical trials. Biometrics, 35(3), 549-556.
        * Pocock, S. J. (1977). Group sequential methods in the design
          and analysis of clinical trials. Biometrika, 64(2), 191-199.
        * David L. Demets, K. K. Gordon Lan (1994). Interim analysis:
          The alpha spending function approach. Statistics in Medicine.
        * Jennison C., and Turnbull B. W (2000). Group Sequential Methods
          with Applications to Clinical Trials. Boca Raton, Florida:
          Chapman & Hall/CRC.
    """
    if iuse == 1:
        name = "O'Brien-Fleming"
        spending = 2 * (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(t**phi)))
    elif iuse == 2:
        name = "Pocock"
        spending = alpha * np.log(1 + (np.exp(1) - 1) * t)
    elif iuse == 3:
        name = "Kim-DeMets"
        spending = alpha * t**phi
    elif iuse == 4:
        name = "Hwang-Shih-DeCani"
        spending = np.where(
            phi == 0, alpha * t, alpha * (1 - np.exp(-phi * t)) / (1 - np.exp(-phi))
        )
    elif iuse == 5:
        name = "Haybittle-Peto"
        spending = np.where(t < 1, 1 - norm.cdf(3 * phi), alpha)
    else:
        raise ValueError("Invalid iuse value. Must be 1, 2, 3, 4, or 5.")

    return (name, spending)


def multivariate_norm_cdf(
    upper: np.ndarray,
    lower: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    focus: str = "performance",
) -> float:
    """
    Calculates CDF of a multivariate normal distribution.
    """
    with redirect_stdout(io.StringIO()):
        if focus == "performance":
            return mvnormcdf(upper=upper, lower=lower, mu=mean, cov=cov)
        elif focus == "accuracy":
            return multivariate_normal.cdf(
                x=upper, lower_limit=lower, mean=mean, cov=cov
            )
        else:
            raise ValueError(
                "Invalid `focus` value. Must be 'performance' or 'accuracy'."
            )


def calculate_sequential_bounds(
    time_points: np.ndarray,
    alpha: float = 0.05,
    iuse: int = 1,
    phi: float = 1,
    beta: float = None,
    tol: float = 1e-05,
    binding: bool = False,
    beta_iuse: int = None,
    beta_phi: float = None,
) -> tuple:
    """
    Calculates the upper and lower bounds for a sequential design.

    This function computes the stopping boundaries for a sequential design with a
    specified number of stages, alpha (Type I error rate), and beta (Type II error rate).
    It uses the brentq root-finding algorithm and multivariate normal CDF calculations.

    1. If statistic value exceed upper bound - the null hypothesis is rejected
        and superiority is declared.
    2. If statistic value is below the lower bound - the null hypothesis is not
        rejected and an experiment is stopped for futility.

    Args:
        time_points (list): A vector of information fractions at each interim analysis.
        alpha (float): The overall Type I error rate.
        iuse (int): An indicator of the alpha spending function type
        phi (float): A parameter used for alpha spending function
        beta (float, optional): The overall Type II error rate.
        tol (float, optional): The tolerance for the root-finding algorithm.
        binding (bool, optional): If True binding algorithm in futility bounds calculation is used.
        beta_iuse (int, optional): An indicator of beta spending function type, by default equal to iuse.
        beta_phi (float, optional): A parameter used for beta spending function, by default equal to phi.

    Returns:
        tuple: contains the lower and upper bounds for each stage.

    Reference:
      * Chang MN, Hwang I, Shih WJ. (1998)
        Group sequential designs using both type I and type II error probability
        spending functions. Communications in Statistics - Theory Methods.
        https://www.tandfonline.com/doi/abs/10.1080/03610929808832161
      * Wu, J., & Li, Y. (2020)
        Group sequential design for historical control trials using error
        spending functions. Journal of biopharmaceutical statistics, 30(2), 351–363.
        https://europepmc.org/article/MED/31718458#free-full-text
      * Pampallona, S., Tsiatis, A.A., Kim, K.M. (2001)
        Interim Monitoring of Group Sequential Trials Using Spending Functions
        for the Type I and Type II Error Probabilities. Drug Information Journal 35:1113-1121
        https://sci-hub.se/10.1177/009286150103500408
      * Georgi Z. Georgiev (2017)
        Efficient A/B Testing in Conversion Rate Optimization: The AGILE Statistical Method
        Analytics-Toolkit.com (binding and non binding futility bounds)
        https://www.analytics-toolkit.com/pdf/Efficient_AB_Testing_in_Conversion_Rate_Optimization_-_The_AGILE_Statistical_Method_2017.pdf

    Acknowledgments:
      The following implementation besides all the rest was inspired by
          * https://github.com/cran/ldbounds/blob/master/R/landem.R
          * https://github.com/denim-bluu/advanced_ab_test_py/blob/main/src/seq_design/spend_func.py
    """

    def calculate_upper_bound(
        x: float,
        upper_bounds: np.ndarray,
        covariance_matrix: np.ndarray,
        target_probability: float,
        known_lower_bounds: np.ndarray = None,
    ) -> float:
        """Calculates the upper bound at a specific stage."""
        num_bounds = len(upper_bounds)
        if known_lower_bounds is not None and len(known_lower_bounds) == num_bounds:
            # Use futility bounds in binding scenario
            lower_bounds = known_lower_bounds
            focus = "accuracy"
        else:
            lower_bounds = np.full(num_bounds, -np.inf)
            focus = "performance"
        upper = np.concatenate((upper_bounds, [np.inf]))
        lower = np.concatenate((lower_bounds, [x]))
        mean_vector = np.zeros(num_bounds + 1)
        probability = multivariate_norm_cdf(
            upper=upper, lower=lower, mean=mean_vector, cov=covariance_matrix, focus=focus
        )
        return target_probability - probability

    def calculate_lower_bound(
        x: float,
        lower_bounds: np.ndarray,
        eta_mean: float,
        time_points: np.ndarray,
        covariance_matrix: np.ndarray,
        target_probability: float,
    ) -> float:
        """Calculates the lower bound at a specific stage."""
        num_bounds = len(lower_bounds)
        upper_bounds = np.full(num_bounds, np.inf)
        upper = np.concatenate((upper_bounds, [x]))
        lower = np.concatenate((lower_bounds, [-np.inf]))
        mean_vector = eta_mean * np.sqrt(time_points[: num_bounds + 1])
        probability = multivariate_norm_cdf(
            upper=upper, lower=lower, mean=mean_vector, cov=covariance_matrix
        )
        return target_probability - probability

    # Generate time points
    num_stages = len(time_points)

    # Calculate alpha and beta spending functions (assuming this function is defined elsewhere)
    _, alpha_spending = alpha_spending_function(
        time_points, iuse=iuse, phi=phi, alpha=alpha
    )

    # Calculate incremental alpha and beta values
    incremental_alpha = np.concatenate(
        (alpha_spending[:1], alpha_spending[1:] - alpha_spending[:-1])
    )

    # Calculate covariance matrix (vectorized)
    i, j = np.indices((num_stages, num_stages))
    covariance_matrix = np.minimum(time_points[i], time_points[j]) / np.sqrt(
        time_points[i] * time_points[j]
    )

    # Initialize bounds
    upper_bounds = np.zeros(num_stages)
    upper_bounds[0] = norm.ppf(1 - incremental_alpha[0])

    # Calculate upper bounds
    for i in range(1, num_stages):
        if iuse == 5 and i < num_stages - 1:
            upper_bounds[i] = norm.ppf(1 - incremental_alpha[0])
        else:
            args = (
                upper_bounds[:i],
                covariance_matrix[: i + 1, : i + 1],
                incremental_alpha[i],
            )
            upper_bounds[i] = root(calculate_upper_bound, -10, 10, args=args)

    # If beta is not provided, stopping for futility bounds are not calculated
    if not beta:
        return (incremental_alpha, upper_bounds)

    # Apply same processing as for alpha earlier
    name, beta_spending = alpha_spending_function(
        time_points, iuse=beta_iuse or iuse, phi=beta_phi or phi, alpha=beta
    )
    incremental_beta = np.concatenate(
        (beta_spending[:1], beta_spending[1:] - beta_spending[:-1])
    )
    lower_bounds = np.zeros(num_stages)

    # Calculate initial eta values
    eta_0 = norm.ppf(1 - alpha) + norm.ppf(1 - beta)
    eta_1 = np.sqrt(2) * eta_0

    # Loop metadata
    converged = False
    iteration = 0

    # Calculate lower bounds and adjust eta_mean iteratively
    while not converged:

        iteration += 1
        boundary_violation = False  # Flag to check if lower bound exceeds upper bound
        eta_mean = (eta_0 + eta_1) / 2
        lower_bounds[0] = norm.ppf(incremental_beta[0]) + eta_mean * np.sqrt(
            time_points[0]
        )

        # current iteration upper bounds
        # initialized with non-binding calculation result
        current_upper_bounds = upper_bounds.copy()

        if lower_bounds[0] > upper_bounds[0]:
            eta_1 = eta_mean  # Adjust eta_mean if initial lower bound is too high
        else:
            for i in range(1, num_stages):
                if (beta_iuse or iuse) == 5:
                    lower_bounds[i] = norm.ppf(
                        incremental_beta[0]
                    ) + eta_mean * np.sqrt(time_points[i])
                else:
                    if binding:
                        # recalculate upper bound[i], binding to lower bound [:i]
                        args_upper = (
                            current_upper_bounds[:i],
                            covariance_matrix[: i + 1, : i + 1],
                            incremental_alpha[i],
                            lower_bounds[:i]
                        )
                        # print(incremental_alpha[i], '\n' , incremental_alpha, '\n', lower_bounds)
                        try:
                            # attempt to find the binding root
                            current_upper_bounds[i] = root(calculate_upper_bound, -10, 10, args=args_upper)
                        except ValueError as e:
                            # Fallback for the last step, where the algorithm may fail due to alpha exhaustion
                            if i == num_stages - 1:
                                # Final stage fallback: allow failure and use the non-binding bound
                                pass
                            else:
                                raise ValueError(
                                    f"Sequential bounds algorithm failed at interim stage {i + 1}. "
                                    f"The root-finding algorithm (brentq) could not find a boundary "
                                    f"within the search bracket. This is likely due to machinery precision "
                                    f"limits or a highly conservative spending function (like "
                                    f"O'Brien-Fleming) requiring an incremental alpha that is effectively "
                                    f"zero. Original error: {e}"
                                )

                    args_lower = (
                        lower_bounds[:i],
                        eta_mean,
                        time_points,
                        covariance_matrix[: i + 1, : i + 1],
                        incremental_beta[i],
                    )
                    lower_bounds[i] = root(calculate_lower_bound, -10, 10, args=args_lower)
                if lower_bounds[i] > current_upper_bounds[i]: # Use current upper bound for the check
                    eta_1 = (
                        eta_mean  # Adjust eta_mean if lower bound exceeds upper bound
                    )
                    boundary_violation = True
                    break

            if binding and not boundary_violation:
                upper_bounds = current_upper_bounds.copy()

            if not boundary_violation:
                lower_bounds[num_stages - 1] = upper_bounds[
                    num_stages - 1
                ]  # Set last lower bound to last upper bound

                # Calculate cumulative probabilities
                cumulative_probabilities = np.empty_like(lower_bounds)
                cumulative_probabilities[0] = norm.cdf(
                    lower_bounds[0], loc=eta_mean * np.sqrt(time_points[0])
                )
                for i in range(1, num_stages):
                    upper = np.concatenate((upper_bounds[:i], [lower_bounds[i]]))
                    lower = np.concatenate((lower_bounds[:i], [-np.inf]))
                    mean_vector = eta_mean * np.sqrt(time_points[: i + 1])
                    cumulative_probabilities[i] = multivariate_norm_cdf(
                        upper=upper,
                        lower=lower,
                        mean=mean_vector,
                        cov=covariance_matrix[: i + 1, : i + 1],
                    )

                beta_k = sum(
                    cumulative_probabilities
                )  # Calculate overall beta at this stage

                # Adjust eta_mean based on calculated beta_k
                if beta_k < beta:
                    eta_1 = eta_mean
                else:
                    eta_0 = eta_mean

                # Check for convergence
                if abs(beta - beta_k) < tol:
                    converged = True
                else:  # Facilitate convergence
                    tol += tol

    print(
        f"Sequential boundaries {'non-' if not binding else ''}binding algorithm to stop for futility converged to {tol} "
        f"tolerance in {iteration} iterations using {name} spending function."
    )

    return (lower_bounds, upper_bounds)


def ldBounds(t: np.ndarray, iuse: int = 1, phi: float = 1, alpha: float = 0.05) -> dict:
    """
    Calculates Lan-DeMets boundaries for group sequential testing.

    This function implements the Lan-DeMets method for calculating
    boundaries in group sequential designs. It supports various alpha
    spending functions.

    Args:
        t: A vector of information fractions at each interim analysis.
        iuse: An integer indicating the alpha spending function:
              1 = O'Brien-Fleming (O'Brien and Fleming, 1979)
              2 = Pocock (Pocock, 1977)
              3 = Kim-DeMets (Kim-DeMets, 1994)
              4 = Hwang-Shih-DeCani
              5 = Haybittle-Peto
        phi: A numeric value of the alpha spending function parameter.
        alpha: Overall significance level. Must be between 0 and 1.

    Returns:
        A dictionary contains the upper bounds for each interim analysis.

    References:
      * K. K. Gordon Lan and David L. DeMets (1983)
        Discrete Sequential Boundaries for Clinical Trials
        https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/LanDeMets.pdf
      * K. M. Kim and A. A. Tsiatis (2020)
        Independent increments in group sequential tests: a review
        https://www.idescat.cat/sort/sort442/44.2.1.Kim-Tsiatis.pdf
      * D. Lakens, F. Pahlke, and G. Wassmer (2021)
        Group Sequential Designs: A Tutorial
        https://osf.io/preprints/psyarxiv/x4azm
    """

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if iuse not in [1, 2, 3, 4, 5]:
        raise ValueError("Invalid iuse value. Must be 1, 2, 3, 4, or 5.")
    if iuse == 1 and phi < 1:
        raise ValueError(
            "Phi must be at least 1 for O'Brien-Fleming spending function."
        )
    if iuse == 3 and phi <= 0:
        raise ValueError("Phi must be positive for Kim-DeMets spending function.")
    if iuse == 4 and phi < 0:
        raise ValueError(
            "Phi must be non-negative for Hwang-Shih-DeCani spending function."
        )
    if iuse == 5 and phi < 1:
        raise ValueError(
            "Haybittle-Peto itermittent analyses have to be conservative, use phi >= 1 to set `3 * phi` as a critical value"
        )

    alpha_spending, ubnd = calculate_sequential_bounds(
        time_points=t, alpha=alpha, iuse=iuse, phi=phi
    )

    # for two-sided it must include + norm.cdf(lbnd)
    nominal_alpha = 1 - norm.cdf(ubnd)

    return {
        "time.points": t,
        "alpha.spending": alpha_spending,
        "overall.alpha": np.sum(alpha_spending),
        "upper.bounds": ubnd,
        "nominal.alpha": nominal_alpha,
    }


def GST(
    actual: Union[int, np.ndarray],
    expected: Union[int, np.ndarray],
    iuse: int = 1,
    phi: float = 1,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Calculates GST upper bounds for peeking strategy that differs from expectations.

    This function calculates Lan-DeMets boundaries for group sequential
    testing (GST) when the expected peeking strategy was slightly changed,
    fro example the data is over/under-sampled, meaning interim analyses
    are conducted more/less frequently than initially planned. It adjusts the
    boundary calculation to account for the actual number of analyses.

    Args:
        actual: The actual number or the vector of interim analyses performed.
        expected: The initially planned number or the vector of interim analyses.
        iuse: An integer indicating the alpha spending function.
        phi: A numeric value of the alpha spending function parameter.
        alpha: Overall significance level. Defaults to 0.05.

    Returns:
        A list containing the adjusted GST upper bounds.

    References:
      * G. Wassmer and  W. Brannath (2016)
        Group Sequential and Confirmatory Adaptive Designs in Clinical Trials,
        Chapter 3.3 The alpha-Spending Function Approach, pages 77–79
        https://link.springer.com/book/10.1007/978-3-319-32562-0
    """

    if type(actual) != type(expected):
        raise ValueError(
            "Actual and Expected strategies must be of the same type: "
            "either both vectors or numbers for uniform peeking staretgy"
        )
    elif isinstance(expected, int):
        # round to 5 decimals, machinery precision may break the comparisons
        actual = (1 / expected * np.arange(1, 1 + actual)).round(5)
        expected = np.linspace(1 / expected, 1, expected).round(5)
    else:
        if not all(0 <= i <= 1 for i in expected):  # actual may have higher value
            raise ValueError("Information fractions (t) must be between 0 and 1.")
        if not np.max(expected) == 1:  # actual may have lower or higher value
            raise ValueError("Information fractions (t) must sum to 1.")

    relevant = ldBounds(t=expected, iuse=iuse, phi=phi, alpha=alpha)
    bounds = relevant["upper.bounds"]
    spending = relevant["alpha.spending"]

    def calculate_upper_bound(
        x: float,
        upper_bounds: np.ndarray,
        covariance_matrix: np.ndarray,
        target_probability: float,
    ) -> float:
        """Calculates the upper bound at a specific stage."""
        num_bounds = len(upper_bounds)
        lower_bounds = np.full(num_bounds, -np.inf)
        upper = np.concatenate((upper_bounds, [np.inf]))
        lower = np.concatenate((lower_bounds, [x]))
        mean_vector = np.zeros(num_bounds + 1)
        probability = multivariate_norm_cdf(
            upper=upper, lower=lower, mean=mean_vector, cov=covariance_matrix
        )
        return target_probability - probability

    # wrong experiment design
    if actual[0] != expected[0]:
        raise ValueError("Actual and expected must have at least the same beginning.")

    # exact match of actual and expected peeking points
    elif actual.size == expected.size and all(actual == expected):
        return bounds

    # oversampling - naive over-sample trick suggested in literature
    elif actual.size > expected.size and all(actual[: expected.size] == expected):
        oversampled_bounds = []
        for time in range(expected.size + 1, actual.size + 1):
            oversampled_bounds.append(
                ldBounds(
                    t=actual[:time] / np.max(actual[:time]),
                    iuse=iuse,
                    phi=phi,
                    alpha=alpha,
                )["upper.bounds"][-1]
            )
        bounds = np.concatenate((bounds, oversampled_bounds))
        return bounds

    # undersampling - procedure from the reference book
    elif actual.size < expected.size and all(expected[: actual.size] == actual):

        bounds = bounds[: actual.size - 1]
        time_points = actual / np.max(actual)

        i, j = np.indices((actual.size, actual.size))
        covariance_matrix = np.minimum(time_points[i], time_points[j]) / np.sqrt(
            time_points[i] * time_points[j]
        )

        args = (bounds, covariance_matrix, np.sum(spending[actual.size :]))
        bounds = np.append(bounds, root(calculate_upper_bound, -10, 10, args=args))
        return bounds

    elif actual.size != expected.size:
        # number of intermittent analyses and their timing during data collection needn't to be predetermined
        # https://www.routledge.com/Group-Sequential-Methods-with-Applications-to-Clinical-Trials/Jennison-Turnbull/p/book/9780849303166
        raise ValueError(
            "The function doesn't handle well such massive deviation from expectations"
        )

    else:  # procedure from the reference book

        idx = np.argmin(actual[: expected.size] == expected[: actual.size])
        bounds = bounds[:idx]
        time_points = actual / np.max(actual)

        for time in range(idx, len(actual)):

            i, j = np.indices((time + 1, time + 1))
            covariance_matrix = np.minimum(time_points[i], time_points[j]) / np.sqrt(
                time_points[i] * time_points[j]
            )

            args = (bounds, covariance_matrix, spending[time])
            bounds = np.append(bounds, root(calculate_upper_bound, -10, 10, args=args))

        return bounds
