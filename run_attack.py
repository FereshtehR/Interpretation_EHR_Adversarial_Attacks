# Import necessary libraries
import numpy as np
import torch
import statistics
from utils import calc_criteria, create_weights, l2_distance
from interpretaion_attack import CW  # Assuming you've switched to using the "cw" module
from utils import (
    VisitSequenceWithLabelDataset,
    read_data,
    _load_dataset,
    print_out,
)
from settings import settings

# Disable cuDNN for deterministic results
torch.backends.cudnn.enabled = False

# Check if CUDA is available and if specified to use it
use_cuda = True if (settings.use_cuda and torch.cuda.is_available()) else False
device = torch.device("cuda" if use_cuda else "cpu")

# Create the target content
def create_target_cont(x, one_len, model):
    t_len = [one_len]
    output_neg, org_alpha_neg, org_beta_neg = model(x, t_len)
    W_emb = model.get_Wemb()
    W = model.get_W()
    cont_org = org_alpha_neg * torch.matmul(W, org_beta_neg.transpose(0, 1).transpose(1, 2) * W_emb).transpose(0, 1)[1] * x
    target_cont = cont_org
    return target_cont, output_neg

# Calculate various measurements
def calculate_measurements(
    all_advs, all_cleans, all_adv_conts, all_clean_conts,
    all_adv_labels, all_clean_labels, passed_indices,
    target_data, target_cont, weight_contrs_over_time, model
):
    # Initialize variables
    successes = 0
    length_ = [48]

    all_advs_np = []
    adv_inds = []
    adv_good_look = []
    L1_contr_adv_org_avg, W_contr_adv_org_avg, L1_contr_adv_target_avg, W_contr_adv_target_avg, L1_adv_org_avg, W_adv_org_avg = [], [], [], [], [], []

    second_metrics = [[], [], [], [], [], [], [], [], [], []]
    second_metrics_sum = list()

    # Loop through the passed indices
    for iter, ind in enumerate(passed_indices):
        adv_images = all_advs[iter]
        ehr_input = all_cleans[iter]
        advCont = all_adv_conts[iter]
        cleanCont = all_clean_conts[iter]
        ehr_label = all_clean_labels[iter]

        # Calculate L2 distance
        l2, inds_adversarial = l2_distance(model, ehr_input, adv_images, ehr_label, length_, device=device)

        # Check for successful attacks
        if len(inds_adversarial[0]) != 0:
            print("Successful Attack")
            successes += 1
        else:
            advCont_np = advCont.detach().cpu().numpy()
            cleanCont_np = cleanCont.detach().cpu().numpy()
            target_cont_np = target_cont.detach().cpu().numpy()

            all_advs_np.append(adv_images.cpu().numpy())
            adv_inds.append(ind)

            L1_contr_adv_org, W_contr_adv_org, L1_contr_adv_target, W_contr_adv_target, L1_adv_org, W_adv_org = \
                print_out(advCont, cleanCont, target_cont, adv_images, ehr_input)

            L1_contr_adv_org_avg.append(L1_contr_adv_org)
            W_contr_adv_org_avg.append(W_contr_adv_org)
            L1_contr_adv_target_avg.append(L1_contr_adv_target)
            W_contr_adv_target_avg.append(W_contr_adv_target)
            L1_adv_org_avg.append(L1_adv_org)
            W_adv_org_avg.append(W_adv_org)

            if W_contr_adv_org > 1 * W_contr_adv_target:
                adv_good_look.append(ind)

            criterias = calc_criteria(cleanCont_np, advCont_np, target_cont_np, weight_contrs_over_time, time_K=3, total_K=10)

            for ci, crit in enumerate(criterias):
                second_metrics[ci].append(crit)

    # Calculate average values
    L1_contr_adv_org_avg = np.sum(L1_contr_adv_org_avg) / len(all_advs_np)
    W_contr_adv_org_avg = np.sum(W_contr_adv_org_avg) / len(all_advs_np)
    L1_contr_adv_target_avg = np.sum(L1_contr_adv_target_avg) / len(all_advs_np)
    W_contr_adv_target_avg = np.sum(W_contr_adv_target_avg) / len(all_advs_np)
    L1_adv_org_med = np.median(L1_adv_org_avg)
    L1_adv_org_avg = np.sum(L1_adv_org_avg) / len(all_advs_np)
    W_adv_org_avg = np.sum(W_adv_org_avg) / len(all_advs_np)

    print('************** Average results: ***************')
    print(f"distance to the original sample::L1: {L1_contr_adv_org_avg:0.2f},Wasserstein: {W_contr_adv_org_avg:0.4f}")
    print(f"distance to the target sample: L1: {L1_contr_adv_target_avg:0.2f}, Wasserstein: {W_contr_adv_target_avg:0.4f}")
    print(f"distance to the original sample in input space: L1: {L1_adv_org_avg:0.2f}, L1 median: {L1_adv_org_med:0.2f}, Wasserstein: {W_adv_org_avg:0.4f}")

    # Calculate median and average metrics over all adversarial examples
    second_metrics_sum_median = []
    for i in range(len(second_metrics)):
        aa = sum(second_metrics[i]) / len(second_metrics[i])
        aa_med = statistics.median(second_metrics[i])
        second_metrics_sum.append(aa)
        second_metrics_sum_median.append(aa_med)

    second_metrics_sum = tuple(second_metrics_sum)
    adv_clean_topK_sign_avg, adv_clean_topk2D_avg, adv_clean_wass2D_avg, \
    adv_target_topK_sign_avg, adv_target_topk2D_avg, adv_target_wass2D_avg, \
    target_clean_topK_sign_avg, target_clean_topk2D_avg, target_clean_wass2D_avg, target_clean_wass1D_avg = second_metrics_sum

    print('----- Average values: -----')
    print(f'adv to clean.... topK-2D: {adv_clean_topk2D_avg:0.2f}, topK_sign: {adv_clean_topK_sign_avg:0.2f}, wass-2D: {adv_clean_wass2D_avg:0.2f}')
    print(f'adv to target.... topK-2D: {adv_target_topk2D_avg:0.2f}, topK_sign: {adv_target_topK_sign_avg:0.2f}, wass-2D: {adv_target_wass2D_avg:0.2f}')
    print(f'd(adv_target)/d(adv_clean).... topK-2D: {target_clean_topk2D_avg:0.2f}, topK_sign: {target_clean_topK_sign_avg:0.2f}, wass-2D: {target_clean_wass2D_avg:0.2f}, wass-1D: {target_clean_wass1D_avg:0.2f}')

    second_metrics_sum_median = tuple(second_metrics_sum_median)
    adv_clean_topK_sign_med, adv_clean_topk2D_med, adv_clean_wass2D_med, \
    adv_target_topK_sign_med, adv_target_topk2D_med, adv_target_wass2D_med, \
    target_clean_topK_sign_med, target_clean_topk2D_med, target_clean_wass2D_med, target_clean_wass1D_med = second_metrics_sum_median

    print('----- Median values: -----')
    print(f'adv to clean.... topK-2D: {adv_clean_topk2D_med:0.2f}, topK_sign: {adv_clean_topK_sign_med:0.2f}, wass-2D: {adv_clean_wass2D_med:0.2f}')
    print(f'adv to target.... topK-2D: {adv_target_topk2D_med:0.2f}, topK_sign: {adv_target_topK_sign_med:0.2f}, wass-2D: {adv_target_wass2D_med:0.2f}')
    print(f'd(adv_target)/d(adv_clean).... topK-2D: {target_clean_topk2D_med:0.2f}, topK_sign: {target_clean_topK_sign_med:0.2f}, wass-2D: {target_clean_wass2D_med:0.2f}, wass-1D: {target_clean_wass1D_med:0.2f}')

    all_advs_np = np.concatenate(all_advs_np, axis=0)
    all_advs_np = np.squeeze(all_advs_np)
    adv_inds = np.array(adv_inds)
    adv_groundtruth_labels = np.concatenate([[1] * len(adv_inds)])
    print(f"Success Rate: {successes / len(passed_indices) * 100}")
    print(adv_inds)
    return all_advs_np, adv_inds, adv_groundtruth_labels, adv_clean_topK_sign_med, target_clean_topk2D_med, target_clean_wass2D_med, target_clean_wass1D_med

# Craft adversarial examples
def craft_adversarial_examples(model, target_data, target_cont, d_target_cont, ehr_inputs, ehr_labels, lengths, indices, folder, weight_contrs_over_time, gamma, attack_type):
    atk = CW(model, c=1, kappa=0, steps=1001, lr=0.01, device=device, gamma=gamma, attack_type=attack_type)
    torch.set_printoptions(profile="full", precision=2, linewidth=400)
    np.set_printoptions(threshold=10000, precision=2, linewidth=4000, suppress=True)

    passed_indices = []
    all_advs = []
    all_cleans = []
    all_adv_conts = []
    all_clean_conts = []
    all_adv_labels = []
    all_clean_labels = []

    ### Repeat adversarial generating for 120 examples
    total_adv_num = 120
    for iter, ind in enumerate(range(0, total_adv_num)):
        if iter == 17:
            continue

        ehr_input = ehr_inputs[ind].unsqueeze(0)
        ehr_label = ehr_labels[ind].unsqueeze(0)
        length_ = [lengths[ind]]
        ind_ = indices[ind]

        output_c, _, _ = model(ehr_input, length_)

        if (output_c.max(1)[1]) != ehr_label:
            continue

        passed_indices.append(ind)

        ### Compute purtabations for EHR data
        adv_images, cleanCont, advCont = atk(ehr_input, ehr_label, length_, ind_, target_cont, d_target_cont)
        output_adv, _, _ = model(adv_images, length_)
        all_advs.append(adv_images)
        all_cleans.append(ehr_input)
        all_adv_conts.append(advCont)
        all_clean_conts.append(cleanCont)
        all_adv_labels.append(output_adv)
        all_clean_labels.append(ehr_label)

    all_advs_np, adv_inds, adv_groundtruth_labels, adv_clean_topK_sign_med, target_clean_topk2D_med, target_clean_wass2D_med, target_clean_wass1D_med = calculate_measurements(all_advs, all_cleans, all_adv_conts, all_clean_conts, all_adv_labels, all_clean_labels, passed_indices, target_data, target_cont, weight_contrs_over_time, model)
    return all_advs_np, adv_inds, adv_groundtruth_labels, adv_clean_topK_sign_med, target_clean_topk2D_med, target_clean_wass2D_med, target_clean_wass1D_med


def main(settings):

    folder = 'this'

    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --no-cuda")

    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(settings.seed)

    # Read data
    test_seqs, test_labels = read_data(start=0.8, end=1.0)
    source_x = []
    source_y = []
    target_x = []
    target_y = []
    data_len = test_seqs.shape[0]
    for i in range(data_len):
        if test_labels[i] == 0:  # if source
            source_x.append(test_seqs[i])
            source_y.append(test_labels[i])
        else:  # if target
            target_x.append(test_seqs[i])
            target_y.append(test_labels[i])
    source_x = np.array(source_x)
    source_y = np.array(source_y)
    target_x = np.array(target_x)
    target_y = np.array(target_y)
    test_pos_seqs, test_pos_labels = target_x, target_y
    test_neg_seqs, test_neg_labels = source_x, source_y

    clean_test_set_neg_class = VisitSequenceWithLabelDataset(test_neg_seqs, test_neg_labels, reverse=True)
    clean_test_set_pos_class = VisitSequenceWithLabelDataset(test_pos_seqs, test_pos_labels, reverse=True)

    source_model = torch.load(settings.model_path, map_location=device)
    source_model = source_model.to(device)
    model = source_model.eval()

    ehr_inputs_neg, ehr_labels_neg, lengths_neg, indices_neg = _load_dataset(clean_test_set_neg_class, 1000)
    ehr_inputs_pos, ehr_labels_pos, lengths_pos, indices_pos = _load_dataset(clean_test_set_pos_class, 1000)

    if device.type == 'cuda':
        ehr_inputs_neg = ehr_inputs_neg.cuda()
        ehr_labels_neg = ehr_labels_neg.cuda()
        ehr_inputs_pos = ehr_inputs_pos.cuda()
        ehr_labels_pos = ehr_labels_pos.cuda()

    one_len = lengths_neg[0]

    ### Genearate Target Interpretation for the adversarial examples to mimic it
    target_ind = 1
    target_input = ehr_inputs_neg[target_ind].unsqueeze(0).to(device)
    target_label = ehr_labels_neg[target_ind].unsqueeze(0).to(device)
    if (target_label != 0):
        print('============ bad target ==================')
        exit(0)

    target_input = target_input.detach()
    target_input_var = torch.autograd.Variable(target_input.clone(), requires_grad=True)
    target_cont_in_graph, _ = create_target_cont(target_input_var, one_len, model)
    target_cont = target_cont_in_graph.detach()
    d_target_cont = target_cont

    weight_contrs_over_time = create_weights(ehr_inputs_pos, one_len, model)

    ### craft the adversarial example based on the target interpretation from the EHR Positive input data
    craft_adversarial_examples(model, target_input, target_cont, d_target_cont, ehr_inputs_pos, ehr_labels_pos, lengths_pos, indices_pos, folder, weight_contrs_over_time, settings.gamma, settings.attack_type)

if __name__ == '__main__':
    if settings.threads == -1:
        settings.threads = torch.multiprocessing.cpu_count() - 1 or 1

    print(f'----------------- Step : {settings.step} ------------------')


    adv_clean_topK_sign_med, target_clean_topk2D_med, target_clean_wass2D_med, target_clean_wass1D_med = main(settings)

    ### Log the data if state == 1
    if int(settings.step) == 1:
        with open("log_attacks.txt", "a") as file1:
            file1.write(f'{adv_clean_topK_sign_med:0.2f},{target_clean_topk2D_med:0.2f},{target_clean_wass2D_med:0.2f},{target_clean_wass1D_med:0.2f}\n')
