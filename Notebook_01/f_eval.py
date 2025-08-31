
# ==============================================================================
#                     MODEL EVALUATION FUNCTION
# ==============================================================================
from scipy.stats import pearsonr
import torch

def evaluate_model(f_model, w_model, x_test, y_test, device='cpu'):
    """
    Evaluates the performance of the predictor and conviction networks on the test set.

    Args:
        f_model (nn.Module): The PredictorNet instance.
        w_model (nn.Module): The ConvictionNet instance (can be None if only f_model is trained).
        x_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test target data.
        device (str): Device to evaluate on ('cuda' or 'cpu').

    Returns:
        dict: Evaluation results (Test MSE for f(x), correlation for w(x), etc.).
    """
    # Ensure models are on the correct device and in evaluation mode
    f_model.to(device)
    f_model.eval()
    if w_model is not None:
        w_model.to(device)
        w_model.eval()

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    results = {}

    with torch.no_grad():
        # Get predictions from f_model on x_test
        f_test_preds = f_model(x_test)

        # Calculate Test MSE for f(x)
        test_mse_f = torch.mean((f_test_preds - y_test)**2).item()
        results['test_mse_f'] = test_mse_f
        print(f"Predictor f(x) Test MSE: {test_mse_f:.4f}")

        if w_model is not None:
            # Get predictions from w_model on x_test
            w_test_preds = w_model(x_test)

            # Calculate per-sample squared errors for f(x)
            l_test_actual = (f_test_preds - y_test)**2

            # Flatten w_model predictions and squared errors and move to CPU for numpy conversion
            uncertainty_scores = 1 - w_test_preds.cpu().numpy().flatten()
            actual_errors_flat = l_test_actual.cpu().numpy().flatten()

            # Calculate Pearson correlation and p-value
            correlation, p_value = pearsonr(uncertainty_scores, actual_errors_flat)
            results['correlation_uncertainty_error'] = correlation
            results['p_value_correlation'] = p_value
            print(f"Correlation between uncertainty (1 - w(x)) and actual squared error: {correlation:.4f} (p-value: {p_value:.4f})")

            # Print status message based on correlation
            if correlation > 0.4 and p_value < 0.05:
                print("STATUS: Strong positive correlation suggests w(x) is capturing uncertainty as expected.")
            else:
                print("STATUS: Correlation is weak or not statistically significant. Model may need more training or tuning.")
            results['w_test_preds'] = w_test_preds#.cpu().numpy()
            results['l_test_actual'] = l_test_actual#.cpu().numpy()


        results['f_test_preds'] = f_test_preds#.cpu().numpy()
        results['y_test'] = y_test#.cpu().numpy()
        results['x_test'] = x_test#.cpu().numpy()


    return results
