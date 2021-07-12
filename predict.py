from Utils.LoadData import load_propensity
import torch
import numpy as np
def click():
    click = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.savetxt("../NetWorkResult/w=0.1_relevance=9_author/Click.txt", click, fmt='%.18f')
def predict(model_path, feature_path, propsensity_path, topK_position):
    model = torch.load(model_path)
    model.eval()

    context_feature = np.load(feature_path)
    how_many_query = context_feature.shape[0]
    click = np.zeros(shape=(how_many_query, topK_position, topK_position))
    non_click = np.zeros(shape=(how_many_query, topK_position, topK_position))
    y = load_propensity(propsensity_path)

    context_feature = torch.tensor(context_feature, dtype=torch.float32)
    click = torch.tensor(click, dtype=torch.float32)
    non_click = torch.tensor(non_click, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    with torch.no_grad():
        # loss, err, predict_prop, symmetric_relevance, propensity_layer4, examination_relevance = model(context_feature, y, click, non_click)
        loss, err, predict_prop = model(context_feature, y, click, non_click)
        np.savetxt("../NetWorkResult/w=0.3_relevance=9_author/CPBM.txt", predict_prop, fmt='%.18f')
        # np.savetxt("../NetWorkResult/w=0.6_relevance=9_author/CPBM_train_with_rel_relevance.txt", symmetric_relevance[1], fmt='%.18f')
        # np.savetxt("../NetWorkResult/w=0.6_relevance=9_author/CPBM_train_with_rel_prepensity4.txt", propensity_layer4, fmt='%.18f')
        # np.savetxt("../NetWorkResult/w=0.6_relevance=9_author/CPBM_train_with_rel_examination_relevance.txt", examination_relevance[1], fmt='%.18f')
    # predict_propensity = torch.div(predict_propensity, torch.reshape(predict_propensity[:, 0], (-1, 1)))

    error = torch.sum(torch.abs((y - predict_prop) / y))

    print(error)
# def read_click(file_path):
#     click_log, not_click_log = np.load(file_path)
#     np.savetxt("../NetWorkResult/w=0.6_relevance=9_author/CPBM_train_with_rel_click.txt", click_log[1], fmt='%.18f')
#     np.savetxt("../NetWorkResult/w=0.6_relevance=9_author/CPBM_train_with_rel_not_click.txt", not_click_log[1], fmt='%.18f')
if __name__ == "__main__":
    # topK_position = 10
    #
    # model_para_path = "../NetWorkResult/w=0.3_relevance=9_author/CPBM_model_w=0.3.pkt"
    # test_feature_path = "../Data/trainData/w=0.3_relevance=9_author/train_slice_feature.npy"
    # true_test_path = "../Data/trainData/w=0.3_relevance=9_author/train_slice_propensity.txt"
    #
    # predict(model_para_path, test_feature_path,
    #         true_test_path,
    #         topK_position)
    click()
    # read_click("../Data/trainData/w=0.6_relevance=9_author/train_click.npy")