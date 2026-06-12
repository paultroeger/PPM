"""
Side-by-side illustration of Normal SGD vs DP-SGD workflows.
Uses a simple model on a small random dataset to keep it readable.

Normal SGD backprop works like this:

    1. Compute loss
    2. Compute gradients for each parameter
    3. Update parameters with those gradients

DP-SGD adds two steps between 2 and 3:

    1. Compute loss
    2. Compute per-sample gradients
    3. Clip each per-sample gradient to max norm C (this bounds sensitivity)
    4. Add Gaussian noise to the clipped gradients (this adds privacy)
    5. Update parameters with the noisy gradients

Opacus does steps 3. and 4. for us by wrapping the optimizier in differential privacy
"""

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from test_load_data import get_mnist, get_svhn, get_cifar10
from deep_cnn import deep_cnn

# ── Shared setup ────────────────────────────────────────────────────────────



# We use the cross-entropy loss since we have a classification problem with 10 classes in all cases
cross_entropy_loss  = nn.CrossEntropyLoss() 
processing_unit     = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {processing_unit}\n")


# ── Workflow 1: Normal SGD ───────────────────────────────────────────────────

def train_normal_sgd(dataset, step_size, batch_size, epochs):
    print("=" * 50)
    print("WORKFLOW 1: Normal SGD")
    print("=" * 50)

    # to prevent code doublicates we just pass the private = False flag to the dp_sgd version.
    return train_dp_sgd(dataset, step_size, batch_size, epochs, private = False)


# ── Workflow 2: DP-SGD ───────────────────────────────────────────────────────

def train_dp_sgd(dataset, step_size, batch_size, epochs, private = False, C=None, epsilon=None, delta=None):
    print("=" * 50), print("WORKFLOW 2: DP-SGD (via Opacus)"), print("=" * 50)
    
    # load the right dataset
    if dataset == "MNIST":
        train_loader, test_loader = get_mnist(batch_size)
    elif dataset == "SVHN":
        train_loader, test_loader = get_svhn(batch_size)
    elif dataset == "CIFAR10":
        train_loader, test_loader = get_cifar10(batch_size)


    # start the DP-SGD
    model         = deep_cnn(dataset).to(processing_unit) #use the deep_cnn function to pass the right CNN
    optimizer     = torch.optim.SGD(model.parameters(), step_size, momentum=0.9) #the SGD optimizer by torch
    privacy_engine = PrivacyEngine() #loads opacus PrivacyEngine to privatize SGD with later
    privacy_engine = PrivacyEngine()

    #naturally the PrivacyEngine() will overestimate the privacy budget, leading to something like
    # "epsilon := 1 => actual epsilon is 0.97 at the end of the DP-SGD". To make the 0.97 closer to the 1.0 
    # (i.e. to make the actual epsilon at the end of DP-SGD as close to the target epsilon we set as possible)
    # we must pass alpha ranges to the privacy engine.
    privacy_engine.accountant.DEFAULT_ALPHAS = list(range(2, 512)) 

    if private: # private flag. If set to false we just have normal SGD
        # Opacus wraps the model, optimizer, and dataloader
        # This is the only change needed compared to normal SGD.
        # Opacus automatically uses RDP to enforce the target ε,δ are met.
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module         = model,
            optimizer      = optimizer,
            data_loader    = train_loader,
            target_epsilon = epsilon,      # target ε 
            target_delta   = delta,        # target δ 
            epochs         = epochs,       # needed to know total steps
            max_grad_norm  = C,            # set clipping parameter for the gradients C manually
        )


    for epoch in range(epochs):

        # ── Train / Compute Weights ──────────────────────────────────────────────────────

        #model.train() applied Dropout and BatchNorm:
        #       Dropout:   randomly zeros out some neuron (makes fitting more stable to prevent overfit)
        #       BatchNorm: normalizes the batch, so that gradients don't vanish or explode (NaN) during fitting
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(processing_unit), y_batch.to(processing_unit)

            # zero gradient variable
            optimizer.zero_grad()

            # Step 1: Compute loss (identical to normal SGD)
            output = model(x_batch)
            loss   = cross_entropy_loss(output, y_batch)

            # Step 2: Compute per-sample gradients (Opacus intercepts and further processes this in optimizer.step())
            loss.backward()

            # Step 3: Clip each per-sample gradient to max norm C   ┐
            # Step 4: Add Gaussian noise to the clipped gradients   ├ Opacus does all of this
            # Step 5: Update parameters with the noisy gradients    ┘ 
            optimizer.step()



        # ── Test / Compute Accuracy ──────────────────────────────────────────────────────
        # model.train() applies Dropout and BatchNorm:
        #       Dropout:   turned off (see model.train() for what it does exactly)
        #       BatchNorm: uses the "mean of means" and "mean of variances" to normalize the test batch using all
        #                  observed training batches.
        model.eval()
        correct = 0 #initialize correct classifications count as 0
        total   = 0 #initialize total classifications count as 0

        # torch.no_grad() disables pytorches gradient tracker. Normally all gradients are tracked for backpropagation
        # of error through loss.backward(). Since we are in the test phase we dont need to track gradients anymore.
        with torch.no_grad():
            for x, y in test_loader:
                x, y      = x.to(processing_unit), y.to(processing_unit)    #sending x and y to the processing unit.
                output    = model(x)                                        #model prediction on x retunrs a vector in IR^10 with class scores
                predicted = output.argmax(dim=1)                            #out.argmax(dim=1) picks the class with the highest score -> dim =1
                correct  += (predicted == y).sum().item()                   #increment the correct count if the prediciton was correct
                total    += y.size(0)                                       #increment the total count 
        accuracy = correct / total                                          #accuracy is % of correct classifications of all 


        if private:
            # computes the privacy budget currently spent (once epoch is epoch is reached eps_spent = epsilon must hold.
            eps_spent = privacy_engine.get_epsilon(delta) 
            print(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f} | accuracy: {accuracy:.4f} | ε = {eps_spent:.2f}, δ = {delta}")
        else:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f} | accuracy: {accuracy:.4f}")

    print("DP-SGD training complete.\n")
    return model


# ── Run both ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test normal SGD on MNIST
    train_normal_sgd(
        dataset    = "MNIST",
        step_size  = 0.01,
        batch_size = 64,
        epochs     = 0
    )

    # Test DP-SGD on MNIST
    train_dp_sgd(
        dataset    = "MNIST",
        step_size  = 0.01,
        batch_size = 64,
        epochs     = 25,
        private    = True,
        C          = 1.0,
        epsilon    = 4.0,
        delta      = 1e-5
    )