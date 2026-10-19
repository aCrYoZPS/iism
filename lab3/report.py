import analysis


def _verdict(test):
    return "rejected" if test.rejected else "not rejected"


def component_lines(name, sample, mean0, var0):
    est = analysis.estimates(sample)
    mean_test = analysis.mean_test(sample, mean0)
    var_test = analysis.variance_test(sample, var0)
    return [
        f"{name}:",
        f"  M theory {mean0:.4f}, estimate {est.mean:.4f}",
        f"    95% CI ({est.mean_interval[0]:.4f}, {est.mean_interval[1]:.4f})",
        f"    H0: M = {mean0:g}, t = {mean_test.statistic:.3f}, p = {mean_test.p_value:.3f}"
        f" -> {_verdict(mean_test)}",
        f"  D theory {var0:.4f}, estimate {est.variance:.4f}",
        f"    95% CI ({est.variance_interval[0]:.4f}, {est.variance_interval[1]:.4f})",
        f"    H0: D = {var0:g}, z = {var_test.statistic:.3f}, p = {var_test.p_value:.3f}"
        f" -> {_verdict(var_test)}",
        "",
    ]


def correlation_lines(x, y, rho0):
    corr = analysis.correlation(x, y)
    return [
        "Correlation:",
        f"  rho theory {rho0:.4f}, estimate {corr.r:.4f}",
        f"    95% CI ({corr.interval[0]:.4f}, {corr.interval[1]:.4f})",
        f"  H0: rho = 0, t = {corr.t_statistic:.3f}, p = {corr.p_value:.3f}",
        "",
    ]


def independence_lines(title, test):
    return [
        title,
        f"  statistic {test.statistic:.2f}, df {test.dof}",
        f"  critical (95%) {test.critical:.2f}, p = {test.p_value:.3f}",
        f"  -> {_verdict(test)}",
        "",
    ]
