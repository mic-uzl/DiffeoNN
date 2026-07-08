import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from can_reg import get_trans

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

data_folder = "/data"
#test data
test_image_dir = data_folder + "/empi_pairs/test/images"
test_trans_image_dir = data_folder + "/trans_data/test/images"

torch.set_default_dtype(torch.float64)


def test_invariance(n=10, output_dir=None):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print(f"Testing invarinace of canonicalisation with {n} images and corresponding transformed images")
    # lists to store losses
    final_loss_list = []
    first_loss_list = []
    trans_final_loss_list = []
    trans_first_loss_list = []

    adv_weight = 0.001
    vae_weight = 0.01

    # go through all image pairs
    for i in range(n):
        # load fixed and moving images
        print(f"Processing image pair {i+1}/{n}:")
        fix_path = os.path.join(test_image_dir, f"{i+1}" + ".png")
        mov_path = os.path.join(test_trans_image_dir, f"{i+1}" + ".png")
        
        # get fixed and moving images, warped image, new locations, and backward locations
        trans_fixed_img, trans_moved_img, trans_warped_img, trans_new_locs, trans_new_locs_backward, trans_neg_flow, trans_inshape, trans_first_loss, trans_final_loss = get_trans(fix_path=fix_path, mov_path=mov_path, adv_weight = adv_weight, vae_weight_var=vae_weight)
        trans_final_loss_list.append(trans_final_loss)
        trans_first_loss_list.append(trans_first_loss)
        fixed_img, moved_img, warped_img, new_locs, new_locs_backward, neg_flow, inshape, first_loss, final_loss = get_trans(fix_path=mov_path, mov_path=fix_path, adv_weight = adv_weight, vae_weight_var=vae_weight)
        final_loss_list.append(final_loss)
        first_loss_list.append(first_loss)

        #visualise
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.tight_layout()
        plt.title(f'E = {first_loss:.6f}', fontsize=24)
        plt.imshow(moved_img, cmap='viridis', vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(output_dir + f'{i+1}_moving.png')
        plt.close()

        fig, ax = plt.subplots()
        plt.axis('off')
        plt.tight_layout()
        plt.title(f'E = {final_loss:.6f}', fontsize=24)
        plt.imshow(warped_img, cmap='viridis', vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(output_dir + f'{i+1}_warped.png')
        plt.close()

        fig, ax = plt.subplots()
        plt.axis('off')
        plt.tight_layout()
        plt.title(f'E = {trans_first_loss:.6f}', fontsize=24)
        plt.imshow(trans_moved_img, cmap='viridis', vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(output_dir + f'{i+1}_trans_moving.png')
        plt.close()

        fig, ax = plt.subplots()
        plt.axis('off')
        plt.tight_layout()
        plt.title(f'E = {trans_final_loss:.6f}', fontsize=24)
        plt.imshow(trans_warped_img, cmap='viridis', vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(output_dir + f'{i+1}_trans_warped.png')
        plt.close()

    # Convert to arrays
    final_loss_array = np.array(final_loss_list)
    trans_final_loss_array = np.array(trans_final_loss_list)
    first_loss_array = np.array(first_loss_list)
    trans_first_loss_array = np.array(trans_first_loss_list)

    # print average difference
    print(f"Average difference between the energies: {np.mean(abs(final_loss_array - trans_final_loss_array))}")

    # Plot the losses
    fig, ax = plt.subplots()
    plt.title(f'Losses')
    ax.plot(first_loss_array, label="First loss (original)")
    ax.plot(final_loss_array, label="Final loss (original)")
    ax.plot(trans_first_loss_array, label="First loss (trans)")
    ax.plot(trans_final_loss_array, label="Final loss (trans)")
    ax.set_xlabel('Indices')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir + f'losses.png')
    plt.close() 

    # Plot the difference
    fig, ax = plt.subplots()
    plt.title('Final Loss Difference (original - final)')

    ax.plot(final_loss_array - trans_final_loss_array)
    ax.set_xlabel('Indices') 
    ax.set_ylabel('Loss')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir + f'losses_diff.png')
    plt.close() 

    # save the losses to a file
    metrics_file = os.path.join(output_dir, 'losses.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Number of images: {n}\n")
        f.write(f"Average difference between the energies: {np.mean(abs(final_loss_array - trans_final_loss_array))}\n")
        f.write(f"First losses: {first_loss_list}\n")
        f.write(f"Final losses: {final_loss_list}\n")
        f.write(f"First trans losses: {trans_first_loss_list}\n")
        f.write(f"Final trans losses: {trans_final_loss_list}\n")

if __name__ == '__main__':
    n=20 
    output_dir = '/results/invariance_test/'
    test_invariance(n=n, output_dir=output_dir)
