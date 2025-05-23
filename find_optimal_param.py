import os
import subprocess

# default parameters
DATA_N = 50
DATASET_N = 20
CDF_GAUSS_N = 20
CDF_GAMMA_N = 10
CDF_JBETA_N = 40

MU_PRIOR_MU = 0.0
MU_PRIOR_SIGMA = 4.0
SIGMA_PRIOR_PARAM_A = 0.5
SIGMA_PRIOR_PARAM_B = 2.0
SAMPLE_REPEAT = 2000000


def run_command_get_stdout(cmd):
    """
    Run a shell command and return (exit_code, stdout)
    """
    result = subprocess.run(cmd, shell=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout


def run_test(
    data_n: int = DATA_N,
    dataset_n: int = DATASET_N,
    cdf_gauss_n: int = CDF_GAUSS_N,
    cdf_gamma_n: int = CDF_GAMMA_N,
    cdf_jbeta_n: int = CDF_JBETA_N,
    file_name: str = "results/test_result",
    mu_prior_mu: float = MU_PRIOR_MU,
    mu_prior_sigma: float = MU_PRIOR_SIGMA,
    sigma_prior_param_a: float = SIGMA_PRIOR_PARAM_A,
    sigma_prior_param_b: float = SIGMA_PRIOR_PARAM_B,
    sample_repeat: int = SAMPLE_REPEAT,
) -> str:
    """
    Run the test with the given parameters
    """
    ret, stdout = run_command_get_stdout(
        f"./Gaussian_poolOrNot_cli {data_n} "
        f"{dataset_n} {cdf_gauss_n} {cdf_gamma_n} "
        f"{cdf_jbeta_n} {file_name} {mu_prior_mu} {mu_prior_sigma} "
        f"{sigma_prior_param_a} {sigma_prior_param_b} {sample_repeat}"
    )
    if ret != 0:
        print(f"Error: {ret}")
        print(stdout)
        raise Exception("Error in running the test")
    return stdout


def test_of_data_n():
    """
    Test the result of different data n
    """
    save_file = "results/test_of_data_n"
    data_n_list = [25, 50, 100, 200, 400]
    results = []
    for data_n in data_n_list:
        result = run_test(data_n=data_n, file_name=save_file).strip()
        output_str = f"N = {data_n} → {result}"
        print(output_str)
        results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def test_of_cdf_n():
    """
    Test the result of different cdf n
    """
    save_file = "results/test_of_cdf_n"
    cdf_gauss_n_list = [CDF_GAUSS_N // 2, CDF_GAUSS_N, CDF_GAUSS_N * 2]
    cdf_gamma_n_list = [CDF_GAMMA_N // 2, CDF_GAMMA_N, CDF_GAMMA_N * 2]
    cdf_jbeta_n_list = [CDF_JBETA_N // 2, CDF_JBETA_N, CDF_JBETA_N * 2]
    results = []

    # for all permutations of cdf_gauss_n, cdf_gamma_n, cdf_jbeta_n
    for cdf_gauss_n in cdf_gauss_n_list:
        for cdf_gamma_n in cdf_gamma_n_list:
            for cdf_jbeta_n in cdf_jbeta_n_list:
                result = run_test(cdf_gauss_n=cdf_gauss_n,
                                  cdf_gamma_n=cdf_gamma_n,
                                  cdf_jbeta_n=cdf_jbeta_n,
                                  file_name=save_file).strip()
                output_str = (
                    f"(GN, GMN, BN) = ({cdf_gauss_n}, {cdf_gamma_n}, "
                    f"{cdf_jbeta_n}) → {result}"
                )
                print(output_str)
                results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def test_of_mu_prior_mu():
    """
    Test the result of different mu prior mu
    """
    save_file = "results/test_of_mu_prior_mu"
    mu_prior_mu_list = [-100.0, -10.0, 0.0, 10.0, 100.0]
    results = []
    for mu_prior_mu in mu_prior_mu_list:
        result = run_test(mu_prior_mu=mu_prior_mu, file_name=save_file).strip()
        output_str = f"mu_prior_mu = {mu_prior_mu} → {result}"
        print(output_str)
        results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def test_of_mu_prior_sigma():
    """
    Test the result of different mu prior sigma
    """
    save_file = "results/test_of_mu_prior_sigma"
    mu_prior_sigma_list = [0.1, 1, 4, 10, 50, 200, 1000]
    results = []
    for mu_prior_sigma in mu_prior_sigma_list:
        result = run_test(
            mu_prior_sigma=mu_prior_sigma, file_name=save_file
        ).strip()
        output_str = f"mu_prior_sigma = {mu_prior_sigma} → {result}"
        print(output_str)
        results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def test_of_sigma_prior_param_a():
    """
    Test the result of different sigma prior param a
    """
    save_file = "results/test_of_sigma_prior_param_a"
    sigma_prior_param_a_list = [0.5, 1, 2, 5, 10, 100]
    results = []

    for sigma_prior_param_a in sigma_prior_param_a_list:
        result = run_test(
            sigma_prior_param_a=sigma_prior_param_a, file_name=save_file
        ).strip()
        output_str = f"sigma_prior_param_a = {sigma_prior_param_a} → {result}"
        print(output_str)
        results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def test_of_sigma_prior_param_b():
    """
    Test the result of different sigma prior param b
    """
    save_file = "results/test_of_sigma_prior_param_b"
    sigma_prior_param_b_list = [0.01, 0.1, 1, 2, 5, 10]
    results = []
    for sigma_prior_param_b in sigma_prior_param_b_list:
        result = run_test(
            sigma_prior_param_b=sigma_prior_param_b, file_name=save_file
        ).strip()
        output_str = f"sigma_prior_param_b = {sigma_prior_param_b} → {result}"
        print(output_str)
        results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def test_of_sample_repeat():
    """
    Test the result of different sample repeat
    """
    save_file = "results/test_of_sample_repeat"
    sample_repeat_list = [2000, 20000, 200000, 2000000, 20000000, 200000000]
    results = []
    for sample_repeat in sample_repeat_list:
        result = run_test(
            sample_repeat=sample_repeat, file_name=save_file
        ).strip()
        output_str = f"sample_repeat = {sample_repeat} → {result}"
        print(output_str)
        results.append(output_str)

    with open(save_file, "a") as f:
        for result in results:
            f.write(result + "\n")


def optimal_params_test():
    """
    Test the optimal parameters
    """
    data_n = 100
    cdf_gauss_n = 40
    cdf_gamma_n = 20
    cdf_jbeta_n = 40
    mu_prior_mu = 0.0
    mu_prior_sigma = 50.0
    sigma_prior_param_a = 0.5
    sigma_prior_param_b = 2.0
    sample_repeat = 2000000

    save_file = "results/optimal_params_test"
    result = run_test(data_n=data_n,
                      cdf_gauss_n=cdf_gauss_n,
                      cdf_gamma_n=cdf_gamma_n,
                      cdf_jbeta_n=cdf_jbeta_n,
                      mu_prior_mu=mu_prior_mu,
                      mu_prior_sigma=mu_prior_sigma,
                      sigma_prior_param_a=sigma_prior_param_a,
                      sigma_prior_param_b=sigma_prior_param_b,
                      sample_repeat=sample_repeat,
                      file_name=save_file).strip()
    param_str = (
        f"N={data_n}, GN={cdf_gauss_n}, GMN={cdf_gamma_n}, "
        f"BN={cdf_jbeta_n}, mu_mu={mu_prior_mu}, mu_sig={mu_prior_sigma}, "
        f"sig_a={sigma_prior_param_a}, sig_b={sigma_prior_param_b}, "
        f"SN={sample_repeat}"
    )

    result_str = f"Optimal parameters → {result}"
    result_str = f"{param_str}\n{result_str}"
    print(result_str)
    with open(save_file, "a") as f:
        f.write(result_str + "\n")


if __name__ == "__main__":
    """
    Run the test
    """
    os.system("make")
    os.system("mkdir -p results")
    test_of_data_n()
    test_of_cdf_n()
    test_of_mu_prior_mu()
    test_of_mu_prior_sigma()
    test_of_sigma_prior_param_a()
    test_of_sigma_prior_param_b()
    test_of_sample_repeat()
    optimal_params_test()
