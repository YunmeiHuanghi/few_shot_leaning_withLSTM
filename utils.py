import torch

def calculate_accuracy(output, target):
    """
    Calculates the accuracy of the learner's predictions on a batch of images and their corresponding labels.
    
    Args:
    - output: tensor of shape (batch_size, num_classes) containing the learner's predicted logits for each image
    - target: tensor of shape (batch_size) containing the true labels for each image
    
    Returns:
    - accuracy: scalar value between 0 and 1 representing the accuracy of the learner's predictions on the batch
    """
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)
    return accuracy
