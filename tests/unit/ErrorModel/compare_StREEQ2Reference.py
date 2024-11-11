from manufactured_data import *

def get_input(beta):
    modified_params, gamma = initialize_params(dimensionality=2)
    X1 = np.linspace(start=[0.1], stop=[0.01], num=5)
    X2 = np.linspace(start=[0.3], stop=[0.02], num=5)
    X = np.array(np.meshgrid(X1, X2)).T.reshape(-1, gamma.shape[0])
    W = np.ones((X.shape[0],))
    Y = get_true_values(gamma, X, beta)
    p = 2
    return gamma, X, W, Y, p

def get_Y(beta, gamma, X):
    Y = (beta[0] + beta[1] * X[:, 0].flatten() ** gamma[0]
        + beta[2] * X[:, 1].flatten() ** gamma[1]
        + beta[3] * X[:, 0].flatten() ** gamma[0] * X[:, 1].flatten() ** gamma[1])
    return Y

def get_true_values(gamma, X, beta):
    Y = get_Y(beta, gamma, X)
    return Y

def get_comp_values(gamma, X, W, Y, p, beta):
    objective, Yfit, residual = objective_function(beta, X, Y, W, p, gamma)
    return objective, beta, Yfit, residual

def get_diff(objective_StREEQ, beta_StREEQ, Yfit_StREEQ, residual_StREEQ, 
    objective_ref, beta_ref, Yfit_ref, residual_ref):
    object_diff = np.abs(objective_StREEQ - objective_ref)
    beta_diff = np.amax(np.absolute((beta_StREEQ - beta_ref)/beta_ref))
    Yfit_diff = np.amax(np.absolute((Yfit_StREEQ - Yfit_ref)/Yfit_ref))
    residual_diff = np.amax(np.abs(residual_StREEQ - residual_ref))
    return object_diff, beta_diff, Yfit_diff, residual_diff

if __name__ == "__main__":
    run_type = "Upper" # "Lower"
    if run_type == "Upper":
        beta_true = np.array([10, 20, 30, 40, ])
        beta_StREEQ = np.array([9.0, 33.62229102,  45.23296103, -167.50782826])
        beta_ref = np.array([9.0, 33.62229101, 45.23296103, -167.50782812])
    elif run_type == "Lower":
        beta_true = np.array([1, 2, 3, 4, ])
        beta_StREEQ = np.array([5.0, -52.49291457, -57.94250135, 834.17206463])
        beta_ref = np.array([5.0, -52.49291457, -57.94250114, 834.17206253])
    gamma, X, W, Y, p = get_input(beta_true)
    objective_StREEQ, beta_StREEQ, Yfit_StREEQ, residual_StREEQ = get_comp_values(gamma, X, W, Y, p, beta_StREEQ)
    objective_ref, beta_ref, Yfit_ref, residual_ref = get_comp_values(gamma, X, W, Y, p, beta_ref)
    object_diff, beta_diff, Yfit_diff, residual_diff = get_diff(objective_StREEQ, beta_StREEQ, Yfit_StREEQ, residual_StREEQ, 
                                                       objective_ref, beta_ref, Yfit_ref, residual_ref)
    print(f"StREEQ objective: {objective_StREEQ}")
    print(f"Reference objective: {objective_ref}]")
    print(f"Absolute difference: {object_diff}")
    print("")
    print(f"Calculated beta: {beta_StREEQ}")
    print(f"Reference beta: {beta_ref}")
    print(f"Maximum relative difference: {beta_diff}")
    print("")
    print(f"Calculated Yfit: {Yfit_StREEQ}")
    print(f"Reference Yfit: {Yfit_ref}")
    print(f"True Yfit: {Y}")
    print(f"Maximum relative difference {Yfit_diff}")
    print("")
    print(f"Calculated residual: {residual_StREEQ}")
    print(f"Reference residual: {residual_ref}")
    print(f"Absolute difference {residual_diff}")
