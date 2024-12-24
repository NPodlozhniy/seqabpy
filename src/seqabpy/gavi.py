from scipy.special import gammaln, loggamma, xlogy

from seqabpy import *


class AlwaysValidInference:
    """
    Always valid inference tests allow for continuous testing during data
    collection without stopping rule or the number of intermittent analyses
    identified in advance. Several different implementations are available.

    For one sided alternative it checks for positive uplift only.
    """

    def __init__(
        self,
        size: np.ndarray,
        sigma2: float,
        estimate: float,
        alpha: float = 0.05,
        sides: int = 1,
    ) -> None:
        """
        Args:
            size: Sample sizes.
            sigma2: Original Variance.
            estimate: Estimated effect size (absolute difference).
            alpha: Significance level.
            phi: Parameter for the spending function. ~ sample size

        Returns:
            A boolean indicating whether the result is significant.
        """
        self.size = size
        self.sigma2 = sigma2
        if sides not in [1, 2]:
            raise ValueError("Sides must be 1 or 2.")
        else:
            # formulas that are used are created for two-sided experiments
            # so to run one-sided simply multiply alpha by 2
            self.alpha = 2 * alpha / sides
            self.estimate = abs(estimate) if sides == 2 else estimate

    def GAVI(self, phi: float = None) -> np.ndarray:
        """
        Performs Generalized Always Valid Inference (GAVI). EPPO version.

        Args:
            phi: Parameter for the spending function. ~ sample size

        References:
          * Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, Jasjeet Sekhon (2022)
            Time-uniform, nonparametric, nonasymptotic confidence sequences:
            https://arxiv.org/abs/1810.08240
            The method implemented according to Chapter 3, Formulas 14 and 21;
            Used by Eppo https://docs.geteppo.com/statistics/confidence-intervals/statistical-nitty-gritty/#sequential
        """
        if not phi:
            phi = np.max(self.size)
        # variance of the difference in means
        V = 2 * self.sigma2 / self.size
        # formula 21: normalized boundary minimisation using Lambert W_{-1} approximation
        rho = phi / (
            np.log(np.log(np.exp(1) * self.alpha ** (-2))) - 2 * np.log(self.alpha)
        )
        # formula 14: two-sided normal mixture boundary
        uv = np.sqrt(
            (self.size + rho) * np.log((self.size + rho) / (rho * self.alpha**2))
        )
        ci = (
            np.sqrt(V) * uv / np.sqrt(self.size)
        )  # don't forget to normalize mixture dividing by squared size
        is_significant = (self.estimate - ci) > 0
        return is_significant

    def mSPRT(self, phi: float = None) -> np.ndarray:
        """
        Performs Always Valid F-test (mSPRT). Netflix version.

        This function implements the mSPRT method for sequential testing,
        which provides an always-valid F-test.

        Args:
            phi: Parameter for the spending function. ~ 1/relative_effect_size^2

        References:
          * Michael Lindon, Dae Woong Ham, Martin Tingley, Iavor Bojinov (2024)
            Anytime-Valid Linear Models and Regression Adjusted
            Causal Inference in Randomized Experiments:
            https://arxiv.org/abs/2210.08589
            The method implemented according to Chapter 6, Formula 20;
            In addition there is questionable R code in appendix, didn't get it
            though, perhaps it's mostly for liner model sequences.
        """
        rho, n = 1 / 2, 2 * self.size  # we assume that sample sizes are equal
        V = self.sigma2 / (n * rho * (1 - rho))  # variance of the difference in means
        if phi is None:
            phi = self.sigma2 / (self.estimate**2 * rho * (1 - rho))
        # /!\ at spotify they use z2 = sigma2/V, instead of n, which is wrong
        ci = np.sqrt(V) * np.sqrt(
            np.log((phi + n) / (phi * self.alpha**2)) * (phi + n) / (n)
        )
        is_significant = (self.estimate - ci) > 0
        return is_significant

    def StatSig_SPRT(self):
        """
        Second Approach of Statsig's Implementation of Sequential Testing.
        It mSPRT based on the the approach proposed by Zhao et al.

        References:
          * Zhao Z., Liu M., Deb A. (2019)
            Safely and Quickly Deploying New Features with Staged Rollout
            Framework Using Sequential Test and Adaptive Experimental Design:
            https://arxiv.org/pdf/1905.10493
            Implemented according to Chapter 4, Formula 3, and advised tau
            estimation is taken from appendix.
          * Statsig's implementation details:
            https://docs.statsig.com/experiments-plus/sequential-testing/#statsigs-implementation-of-sequential-testing
        """
        # we assume that sample sizes are equal
        V = 2 * self.sigma2 / self.size  # variance of the difference in means
        # /!\ statsig does a mistake misinterpreting n in tau estimation
        # as the total size, while it's one sample size in original paper
        tau = V * norm.ppf(1 - self.alpha) ** 2  # mixing parameter
        # in the paper it's alpha, not alpha/2
        ci = np.sqrt(
            (V * (V + tau) / tau) * (-2 * np.log(self.alpha) - np.log(V / (V + tau)))
        )
        is_significant = (self.estimate - ci) > 0
        return is_significant

    def statsig_alpha_corrected_v1(self, N: int = None) -> np.ndarray:
        """
        Performs the StatSig sequential test.

        This function implements the StatSig v1 sequential testing procedure, which
        adjusts the critical value at each analysis based on the fraction of
        the total sample size reached.

        Args:
            N: The maximum sample size.
        """
        if not N:
            N = np.max(self.size)
        # we assume that sample sizes are equal
        V = 2 * self.sigma2 / self.size  # variance of the difference in means
        zdiff = self.estimate / np.sqrt(V)
        z_crit_alpha = norm.ppf(
            1 - self.alpha / 2
        )  # here it's a plain procedure, no need to inflate alpha
        zcrit_ss = z_crit_alpha / (self.size / N)
        is_significant = zdiff > zcrit_ss
        return is_significant


def sequential_p_value(counts, assignment_probabilities, dirichlet_alpha=None):
    """
    Compute the sequential p-value for given counts and assignment probabilities.

    References:
      * Lindon, Michael, and Alan Malek. (2022)
        Anytime-Valid Inference For Multinomial Count Data.
        In Advances in Neural Information Processing Systems
        https://openreview.net/pdf?id=a4zg0jiuVi
        https://arxiv.org/pdf/2011.03567
        https://netflixtechblog.com/sequential-testing-keeps-the-world-streaming-netflix-part-2-counting-processes-da6805341642

    This was written in continuation of the first article:
      * Michael Lindon, Chris Sanden, Vaché Shirikian (2022)
        Rapid Regression Detection in Software Deployments through Sequential Testing
        28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining
        https://arxiv.org/abs/2205.14762
        https://netflixtechblog.com/sequential-a-b-testing-keeps-the-world-streaming-netflix-part-1-continuous-data-cba6c7ed49df

    Parameters
    ----------
    counts : array like
        The observed counts in each treatment group.
    assignment_probabilities : array like
        The assignment probabilities to each treatment group.
    dirichlet_alpha : array like, optional
        The Dirichlet mixture parameter.

    Returns
    -------
    float
        The sequential p-value.
    """
    counts = np.array(counts)
    assignment_probabilities = np.array(assignment_probabilities)
    if dirichlet_alpha is None:
        dirichlet_alpha = 100 * assignment_probabilities
    else:
        dirichlet_alpha = np.array(dirichlet_alpha)
    lm1 = (
        loggamma(counts.sum() + 1)
        - loggamma(counts + 1).sum()
        + loggamma(dirichlet_alpha.sum())
        - loggamma(dirichlet_alpha).sum()
        + loggamma(dirichlet_alpha + counts).sum()
        - loggamma((dirichlet_alpha + counts).sum())
    )
    lm0 = gammaln(counts.sum() + 1) + np.sum(
        xlogy(counts, assignment_probabilities) - gammaln(counts + 1), axis=-1
    )
    return min(1, np.exp(lm0 - lm1))
