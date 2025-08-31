# ==============================================================================
#                 PLOTS :  Functions for Visualising Data
# ==============================================================================
import matplotlib.pyplot as plt
import torch
import numpy as np
# Visualise the generated data
def plot_data(x_train, y_train, x_test, y_test,title='Generated Data'):
    plt.figure(figsize=(5, 3))
    plt.scatter(x_train.numpy(), y_train.numpy(), label='Training Data', alpha=0.7, s=10)
    plt.scatter(x_test.numpy(), y_test.numpy(), label='Test Data', alpha=0.7, s=10, c='orange')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot various compenent of the L_new loss :

def plot_L_new(history_lnew):
    # Plot training metrics
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.plot(history_lnew['epoch_loss'])
    plt.title(r'$L_{new}$ (Overall loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(history_lnew['numerator'])
    plt.title(r'Numerator of $L_{new}$')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(history_lnew['denominator'])
    plt.title(r'Denominator of $L_{new}$ (sum $w(x)$)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(history_lnew['avg_w'])
    plt.title(r'Average $w(x)$ on training data')
    plt.xlabel('Epoch')
    plt.ylabel('Average w(x)')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(history_lnew['train_mse_f'])
    plt.title(r'Unweighted MSE of $f(x)$ on training data')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_xy_L_new(x_train,y_train,x_test,y_test,f_test_preds,w_test_preds,l_test_actual):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Analysis of Model Trained with L_new Loss', fontsize=16)
    # 1. Plot f(x) predictions vs. true y
    ax = axes[0, 0]
    ax.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), label='Train Data', alpha=0.2, s=10, color='blue')
    ax.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), label='True Test Data', alpha=0.7, s=20, color='green')
    sorted_indices_test = torch.argsort(x_test.squeeze())
    ax.plot(x_test[sorted_indices_test].cpu().numpy(), f_test_preds[sorted_indices_test].cpu().numpy(), label='f(x) Predictions', color='red', linewidth=2)
    ax.set_title('f(x) Predictions vs. True Data'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(); ax.grid(True)

    # 2. Plot w(x) outputs for the test set
    ax = axes[0, 1]
    ax.scatter(x_test[sorted_indices_test].cpu().numpy(), w_test_preds[sorted_indices_test].cpu().numpy(), label='w(x) Conviction Scores', color='purple', s=15)
    ax.set_title('w(x) Conviction Scores on Test Data'); ax.set_xlabel('x'); ax.set_ylabel('w(x)'); ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(True)

    # 3. Plot actual squared error l(f(x), y)
    ax = axes[1, 0]
    ax.scatter(x_test[sorted_indices_test].cpu().numpy(), l_test_actual[sorted_indices_test].cpu().numpy(), label='Actual Squared Error l(f(x),y)', color='orange', s=15)
    ax.set_title('Actual Squared Error of f(x) on Test Data'); ax.set_xlabel('x'); ax.set_ylabel('Squared Error'); ax.set_yscale('log'); ax.legend(); ax.grid(True)

    # 4. Histogram of w(x) values
    ax = axes[1, 1]
    ax.hist(w_test_preds.cpu().numpy(), bins=20, range=(0,1), edgecolor='black')
    ax.set_title('Histogram of w(x) Test Values'); ax.set_xlabel('w(x)'); ax.set_ylabel('Frequency'); ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ==============================================================================
#                     MODEL COMPARISON AND VISUALIZATION FUNCTION
# ==============================================================================

def compare_models_plot(f_lnew_model, w_lnew_model, f_baseline_model, x_train, y_train, x_test, y_test, device='cpu', title_suffix=''):
    """
    Compares the performance of the L_new model (f and w) and a baseline f model
    on the test set and generates a comparison plot.

    Args:
        f_lnew_model (nn.Module): The trained PredictorNet from L_new training.
        w_lnew_model (nn.Module): The trained ConvictionNet from L_new training.
        f_baseline_model (nn.Module): The trained PredictorNet from baseline training (MSE only).
        x_train (torch.Tensor): Training input data (for plotting).
        y_train (torch.Tensor): Training target data (for plotting).
        x_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test target data.
        device (str): Device to evaluate on ('cuda' or 'cpu').
        title_suffix (str): Suffix to add to plot titles (e.g., ' - Dataset V1').
    """
    # Ensure models are on the correct device and in evaluation mode
    f_lnew_model.to(device)
    f_lnew_model.eval()
    if w_lnew_model is not None:
        w_lnew_model.to(device)
        w_lnew_model.eval()
    f_baseline_model.to(device)
    f_baseline_model.eval()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)


    with torch.no_grad():
        # Get predictions from L_new f_model on x_test
        f_test_preds_lnew = f_lnew_model(x_test)
        # Get predictions from L_new w_model on x_test (if available)
        w_test_preds_lnew = w_lnew_model(x_test) if w_lnew_model is not None else None
        # Get predictions from baseline f_model on x_test
        f_test_preds_baseline = f_baseline_model(x_test)

    # Calculate Test MSE for f_model (L_new)
    test_mse_f_lnew = torch.mean((f_test_preds_lnew - y_test)**2).item()
    print(f"Test MSE for f(x) trained with L_new{title_suffix}: {test_mse_f_lnew:.4f}")

    # Calculate Test MSE for f_model (Baseline)
    test_mse_baseline = torch.mean((f_test_preds_baseline - y_test)**2).item()
    print(f"Test MSE for f(x) trained with Baseline MSE{title_suffix}: {test_mse_baseline:.4f}")

    # Calculate actual squared errors for each test point
    l_test_actual_lnew = (f_test_preds_lnew - y_test)**2

    #########################################
    # --- Visualisations ---
    #########################################

    plt.figure(figsize=(15, 8)) # Increased figure size for subplots

    # Convert tensors to numpy for plotting
    x_train_np = x_train.cpu().numpy().flatten()
    y_train_np = y_train.cpu().numpy().flatten()
    x_test_np = x_test.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()
    f_test_preds_lnew_np = f_test_preds_lnew.cpu().numpy().flatten()
    f_test_preds_baseline_np = f_test_preds_baseline.cpu().numpy().flatten()
    l_test_actual_lnew_np = l_test_actual_lnew.cpu().numpy().flatten()
    w_test_preds_lnew_np = w_test_preds_lnew.cpu().numpy().flatten() if w_lnew_model is not None else None


    # Sort test data by x-values for smooth line plot of predictions
    sorted_indices_test = np.argsort(x_test_np)
    x_test_sorted = x_test_np[sorted_indices_test]
    y_test_sorted = y_test_np[sorted_indices_test]
    f_test_preds_lnew_sorted = f_test_preds_lnew_np[sorted_indices_test]
    f_test_preds_baseline_sorted = f_test_preds_baseline_np[sorted_indices_test]
    l_test_actual_lnew_sorted = l_test_actual_lnew_np[sorted_indices_test]
    w_test_preds_lnew_sorted = w_test_preds_lnew_np[sorted_indices_test] if w_lnew_model is not None else None


    # 1. Plot f(x) predictions vs. true y (L_new vs Baseline)
    plt.subplot(2, 2, 1) # 2x2 grid, first subplot
    plt.scatter(x_train_np, y_train_np, label='Train Data', alpha=0.3, s=10, color='blue')
    plt.scatter(x_test_sorted, y_test_sorted, label='True Test Data', alpha=0.7, s=20, color='green')
    plt.plot(x_test_sorted, f_test_preds_lnew_sorted, label=r'$f(x)$ Predictions $(L_{new})$', color='red', linewidth=2)
    plt.plot(x_test_sorted, f_test_preds_baseline_sorted, label=r'$f(x)$ Predictions (Baseline)', color='orange', linestyle='--', linewidth=2)
    plt.title(r'$f(x)$ Predictions vs. true data ($L_{new}$ vs Baseline)' + title_suffix)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 2. Plot w(x) predictions (if w_lnew_model is provided)
    plt.subplot(2, 2, 2) # 2x2 grid, second subplot
    if w_lnew_model is not None:
        plt.scatter(x_test_sorted, w_test_preds_lnew_sorted, label=r'$w(x)$ Conviction Scores $(L_{new})$', alpha=0.7, s=20, color='purple')
        plt.plot(x_test_sorted, w_test_preds_lnew_sorted, label=r'$w(x)$ Conviction Scores $(L_{new})$', color='purple', linewidth=2)
        plt.title(r'$w(x)$ Conviction Scores on Test Data $(L_{new})$' + title_suffix)
        plt.xlabel('x')
        plt.ylabel(r'$w(x)$')
        plt.ylim(-0.05, 1.05) # Conviction is between 0 and 1
        plt.legend()
        plt.grid(True)
    else:
        plt.title('No Conviction Model Trained with L_new' + title_suffix)
        plt.text(0.5, 0.5, 'w(x) model not available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off') # Hide axes if no plot is generated


    # 3. Plot per-sample squared errors (L_new)
    plt.subplot(2, 2, 3) # 2x2 grid, third subplot
    plt.scatter(x_test_sorted, l_test_actual_lnew_sorted, label='Squared Error 'r'$(f(x) - y)^2$ ($L_{new}$)', alpha=0.7, s=20, color='brown')
    plt.plot(x_test_sorted, l_test_actual_lnew_sorted, label='Squared Error 'r'$(f(x) - y)^2$ ($L_{new}$)', color='brown', linewidth=2)
    plt.title(r'Actual Squared Errors on Test Data ($L_{new}$)' + title_suffix)
    plt.xlabel('x')
    plt.ylabel('Squared Error')
    plt.legend()
    plt.grid(True)

    # 4. Histogram of w(x) values (if w_lnew_model is provided)
    plt.subplot(2, 2, 4) # 2x2 grid, fourth subplot
    if w_lnew_model is not None:
        plt.hist(w_test_preds_lnew_np.flatten(), bins=30, color='teal', alpha=0.7)
        plt.title(r'Distribution of $w(x)$ on Test Data ($L_{new}$)' + title_suffix)
        plt.xlabel(r'$w(x)$ Value')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y')
    else:
        plt.title('No Conviction Model Trained with L_new' + title_suffix)
        plt.text(0.5, 0.5, 'w(x) model not available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off') # Hide axes if no plot is generated


    # Ensure proper layout and display
    plt.tight_layout()
    plt.show()