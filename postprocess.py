import torch
import cv2

def mask_to_box(pixel_mask, link_mask, neighbors=8):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], \
                                    dtype=torch.uint8, device=pixel_mask.device)
    for i in range(neighbors):
        link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class[i]
    link_class = torch.zeros([batch_size, mask_height, mask_width], \
                                dtype=torch.uint8, device=pixel_mask.device)
    # for i in range(neighbors):
    #     link_class = link_class | link_neighbors[:,i]
    res = []
    for i in range(batch_size):
        image = pixel_class[i].cpu().numpy()
        _, box, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # import IPython
        # IPython.embed()
        # res.append(cv2.minAreaRect(box))
        res.append(box[0])
    return res