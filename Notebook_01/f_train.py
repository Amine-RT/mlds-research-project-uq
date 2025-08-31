import torch
import torch.nn as nn
import torch.optim as optim
# from f_models import l_new_regression_loss_fn
# ==============================================================================
#                  TRAINING AND LOSS FUNCTIONS
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the L_new loss function
def l_new_regression_loss_fn(f_preds, w_preds, y_true, epsilon=1e-8):
    """
    Calculates the weighted L_new regression loss.

    Args:
        f_preds (torch.Tensor): Predictions from the predictor model f(x).
        w_preds (torch.Tensor): Predictions from the conviction model w(x) [0-1].
        y_true (torch.Tensor): Ground truth target values.
        epsilon (float): Small value to prevent division by zero in the denominator.

    Returns:
        tuple: (L_new loss, numerator, denominator)
    """
    # Ensure inputs are on the same device
    if f_preds.device != w_preds.device or f_preds.device != y_true.device:
         w_preds = w_preds.to(f_preds.device)
         y_true = y_true.to(f_preds.device)

    # Calculate per-sample squared error
    l_i = (f_preds - y_true)**2

    # Ensure w_preds are within a valid range [epsilon, 1] to avoid issues
    w_preds_clipped = torch.clamp(w_preds, epsilon, 1.0)

    # Calculate the weighted numerator
    numerator = torch.sum(w_preds_clipped * l_i)

    # Calculate the denominator (sum of weights)
    denominator = torch.sum(w_preds_clipped)

    # Calculate the L_new loss
    l_new_loss = numerator / (denominator + epsilon) # Add epsilon to denominator too

    return l_new_loss, numerator, denominator


# Main training function refactored
def train_model(f_model, w_model, x_train, y_train, training_strategy='joint', epochs=4000, learning_rate=1e-4, device='cpu'):
    """
    Trains the predictor and/or conviction models based on the specified strategy.

    Args:
        f_model (nn.Module): The PredictorNet instance.
        w_model (nn.Module): The ConvictionNet instance (can be None).
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        training_strategy (str): Training strategy ('joint', 'f_only', 'separate_freeze_w').
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizers.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        dict: Training history (losses, etc.).
    """
    f_model.to(device)
    if w_model is not None:
        w_model.to(device)

    x_train = x_train.to(device)
    y_train = y_train.to(device).float()

    history = {
#        'epoch_loss': [],
#        'numerator': [],
#        'denominator': [],
#        'avg_w': [],
#        'train_mse_f': []
    }

    if training_strategy == 'f_only':
        print("\n--- Training PredictorNet (f(x)) alone with Standard MSE Loss ---")
        optimizer = optim.Adam(f_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        history['train_loss'] = []

        print(f"Starting f(x) only training for {epochs} epochs...")
        for epoch in range(epochs):
            f_model.train()
            outputs = f_model(x_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())

            if (epoch + 1) % (epochs // 10 if epochs >= 1000 else 100) == 0:
                 print(f"Epoch [{epoch+1: >4}/{epochs}], f(x) only MSE: {loss.item():.6f}")
        print("f(x) only training finished.")


    elif training_strategy == 'joint':
        print("\n--- Training f+w Model with Proposed L_new Loss ---")
        # Ensure f_model and w_model are trainable
        for param in f_model.parameters():
            param.requires_grad = True
        for param in w_model.parameters():
            param.requires_grad = True

        params = list(f_model.parameters()) + list(w_model.parameters())
        optimizer = optim.Adam(params, lr=learning_rate)
        history = {'epoch_loss': [], 'numerator': [], 'denominator': [], 'avg_w': [], 'train_mse_f': []}

        # Optimiser for parameters of both networks
        #optimizer = optim.Adam(list(f_model.parameters()) + list(w_model.parameters()), lr=learning_rate)
        print("--- Starting joint training for {} epochs...".format(epochs))
        for epoch in range(epochs):
            f_model.train()
            w_model.train()

            f_outputs = f_model(x_train)
            w_outputs = w_model(x_train)

            loss_val, num_val, den_val = l_new_regression_loss_fn(f_outputs, w_outputs, y_train)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Calculate unweighted MSE for f(x) on training data
            with torch.no_grad():
                current_train_mse_f = torch.mean((f_outputs - y_train)**2).item()

            # Log metrics
            history['epoch_loss'].append(loss_val.item())
            history['numerator'].append(num_val.item())
            history['denominator'].append(den_val.item())
            history['avg_w'].append(w_outputs.mean().item())
            history['train_mse_f'].append(current_train_mse_f)


            if (epoch + 1) % (epochs // 10 if epochs >= 1000 else 100) == 0:
                print(f"Epoch [{epoch+1: >4}/{epochs}], L_new: {loss_val.item():.6f}, "
                      f"Numerator: {num_val.item():.2f}, Denominator: {den_val.item():.2f}, "
                      f"Avg w(x): {w_outputs.mean().item():.4f}, Train MSE_f: {current_train_mse_f:.4f}")
        print("Joint training finished.")

    elif training_strategy == 'w_only':
        print("\n--- Training ConvictionNet (w(x)) alone with L_new Loss (f(x) frozen) ---")
        # Ensure f_model is frozen
        for param in f_model.parameters():
            param.requires_grad = False
        # Ensure w_model is trainable
        for param in w_model.parameters():
             param.requires_grad = True


        optimizer = optim.Adam(w_model.parameters(), lr=learning_rate)

        history = {'epoch_loss': [], 'numerator': [], 'denominator': [], 'avg_w': []}

        print(f"Starting w(x) only training for {epochs} epochs...")
        for epoch in range(epochs):
            f_model.eval() # f_model should be in eval mode when frozen and used for predictions
            w_model.train()

            with torch.no_grad(): # Ensure no gradients are computed for f_model
                f_outputs = f_model(x_train)

            w_outputs = w_model(x_train)

            loss, num_val, den_val = l_new_regression_loss_fn(f_outputs, w_outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history['epoch_loss'].append(loss.item())
            history['numerator'].append(num_val.item())
            history['denominator'].append(den_val.item())
            history['avg_w'].append(w_outputs.mean().item())

            if (epoch + 1) % (epochs // 10 if epochs >= 1000 else 100) == 0:
                print(f"Epoch [{epoch+1: >4}/{epochs}], L_new: {loss.item():.6f}, "
                      f"Numerator: {num_val.item():.2f}, Denominator: {den_val.item():.2f}, "
                      f"Avg w(x): {w_outputs.mean().item():.4f}")
        print("w(x) only training finished.")


    elif training_strategy == 'separate_freeze_w':
        if w_model is None:
             raise ValueError("w_model must be provided for 'separate_freeze_w' training strategy.")

        print("--- Starting separate training: f(x) with MSE, then w(x) with L_new (f(x) frozen) ---")

        # Step 1: Train f(x) with MSE
        print("\n--- Step 1: Training f(x) with MSE ---")
        optimizer_f = optim.Adam(f_model.parameters(), lr=learning_rate)
        criterion_f = nn.MSELoss()

        # Separate history for this step if needed, or log to main history with flags
        f_only_history_step1 = []
        for epoch in range(epochs):
            f_model.train()
            f_outputs = f_model(x_train)
            loss_val_f = criterion_f(f_outputs, y_train)

            optimizer_f.zero_grad()
            loss_val_f.backward()
            optimizer_f.step()
            f_only_history_step1.append(loss_val_f.item())

            if (epoch + 1) % 800 == 0:
                 print(f"  f(x) training - Epoch [{epoch+1: >4}/{epochs}], MSE: {loss_val_f.item():.4f}")
        print("--- Step 1: f(x) training finished. ---")


        # Step 2: Freeze f(x) parameters
        print("\n--- Step 2: Freezing f(x) parameters ---")
        for param in f_model.parameters():
            param.requires_grad = False
        print("--- Step 2: f(x) parameters frozen. ---")


        # Step 3: Train w(x) with L_new (f(x) frozen)
        print("\n--- Step 3: Training w(x) with L_new (f(x) frozen) ---")
        # Optimiser for w_model parameters only
        optimizer_w = optim.Adam(w_model.parameters(), lr=learning_rate)

        # Separate history for this step if needed
        w_only_history_step3 = {
            'epoch_loss': [],
            'numerator': [],
            'denominator': [],
            'avg_w': [],
            'train_mse_f': []
        }

        for epoch in range(epochs):
            # Set models to training mode (f_model is effectively frozen by requires_grad=False)
            f_model.train() # Keep f_model in train mode
            w_model.train()

            # Forward pass - Use the frozen f_model (no_grad is handled by requires_grad=False)
            f_outputs = f_model(x_train)

            # Forward pass for the ConvictionNet
            w_outputs = w_model(x_train)

            # Calculate L_new loss and its components
            loss_val_w, num_val_w, den_val_w = l_new_regression_loss_fn(
                f_outputs, w_outputs, y_train
            )

            # Backward pass and optimisation - ONLY for w_model parameters
            optimizer_w.zero_grad()
            loss_val_w.backward()
            optimizer_w.step()

            # Log metrics for step 3
            w_only_history_step3['epoch_loss'].append(loss_val_w.item())
            w_only_history_step3['numerator'].append(num_val_w.item())
            w_only_history_step3['denominator'].append(den_val_w.item())
            w_only_history_step3['avg_w'].append(w_outputs.mean().item())
            with torch.no_grad():
                current_train_mse_f = torch.mean((f_outputs - y_train)**2).item()
            w_only_history_step3['train_mse_f'].append(current_train_mse_f)


            if (epoch + 1) % 800 == 0: # Print every 800 epochs for w training
                print(f"  w(x) training - Epoch [{epoch+1: >4}/{epochs}], L_new: {loss_val_w.item():.6f}, "
                      f"Numerator: {num_val_w.item():.2f}, Denominator: {den_val_w.item():.2f}, "
                      f"Avg w(x): {w_outputs.mean().item():.4f}")

        print("--- Step 3: w(x) training finished. ---")
        print("\n--- Separate training finished. ---")

        # Combine histories for plotting (optional, depending on desired plots)
        # For now, we return the w_only history from step 3, and f_only history from step 1
        # This part might need adjustment based on what plots the user wants for this strategy
        # For simplicity now, let's return the w_only history from step 3 and the final f_only MSE
        history = w_only_history_step3
        history['f_only_step1_mse_history'] = f_only_history_step1


    else:
        raise ValueError("Invalid training_strategy. Choose 'joint', 'f_only', or 'separate_freeze_w'.")

    return history