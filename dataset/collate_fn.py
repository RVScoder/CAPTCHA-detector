from config import IMAGE_HEIGHT
import torch

# collate function to manage the variable width of images in each batch

def captcha_collate_fn(batch):

    image_tensors, encoded_labels, labels, widths = zip(*batch)

    max_width = max(widths)
    batch_size = len(image_tensors)

    batch_images = torch.zeros(
        batch_size,
        3,
        IMAGE_HEIGHT,
        max_width,
        dtype = torch.float32
    )

    for i, img in enumerate(image_tensors):
        width = img.shape[2]
        batch_images[i, :, :, :width] = img

    targets = torch.cat(encoded_labels) # concatenating all the encoded_labels into a single tensor
    
    label_lengths = torch.tensor(
        [len(label) for label in labels],
        dtype = torch.long
    )

    widths = torch.tensor(widths, dtype=torch.long)

    return (batch_images, targets, label_lengths, labels, widths)