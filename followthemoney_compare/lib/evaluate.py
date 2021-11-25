from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from collections import Counter, defaultdict


def clean_results(results_lst, th):
    judg_dict = {'positive': 1, 'negative': 0}
    ys = [judg_dict[x[0]] for x in results_lst]
    preds = [1 if x[1] > th else 0 for x in results_lst]
    return ys, preds


def summarize_th_accuracy(metric_dict):
    return {k:np.mean(arr) for k, arr in metric_dict.items()}


def evaluate(results_dict):
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    f1s = defaultdict(list)


    for i in range(0,10):
        th = i / 10
        for pid, results_lst in results_dict.items():
            if results_lst and 'positive' in [x[0] for x in results_lst]:
                ys, preds = clean_results(results_lst, th)
                precision, recall, f1, _ = precision_recall_fscore_support(
                        ys,
                        preds,
                        average='binary',
                        zero_division=1)
                f1s[th].append(f1)
                recalls[th].append(recall)
                precisions[th].append(precision)

    th_precision = summarize_th_accuracy(precisions)
    th_recall = summarize_th_accuracy(recalls)
    th_f1 = summarize_th_accuracy(f1s)
    return th_f1, th_precision, th_recall 


if __name__ == '__main__':
    test_dict = {'1aafe38031744a758ed012dd33947d3a': [],
             '1dfd277a4df641c6963b12ce4a8daa90': [('positive',
               0.8036268151801484)],
             '1e9871788901498aab64b607639f5e40': [],
             '221c1f2cfb5540cf8f38ec34d3d063bc': [('positive',
               0.8036268151801484)],
             '24095fe26a4a4544bd40a13ef1edd7fa': [('positive',
               0.8036268151801484)],
             '2612837bb85e4f84867d42ac7c8395d1': [('positive',
               0.8036268151801484)],
             '27400d721259488598004f0da44ec4fc': [('positive',
               0.8036268151801484)],
             '28f4cd0ad54c432385bc02f2902cef6b': [('positive',
               0.8036268151801484)],
             '2992f18eab3745e7bb2b1f06812e71b5': [('positive',
               0.8036268151801484)],
             '2a1709765db94431ad9bf49ba186c97c': [('positive',
               0.8036268151801484)],
             '2f59f24c68224b27ae619b2824e4100f': [('positive',
               0.8036268151801484)],
             '008aacc5899b42aebe8bfe6b73da261a': [],
             '009949fab205418b937940a7c76b4c4d': [('positive',
               0.8036268151801484)],
             '02a4217e2b7d45c89a94c13762676548': [('positive',
               0.8034925150648269)],
             '09f25549319d45829e9e31d16a09a474': [('positive',
               0.8037275008485661)],
             '0e7d3f376f4848798cea871d67ff61c8': [('positive',
               0.8036268151801484)],
             '11760f26dd394c8d8bcf26598d3865eb': [],
             '129bb7ea6fb04633b734bddd2babbd7a': [],
             '368a904673cb4a6ca9dbf5c9841ead05': [('positive',
               0.8036268151801484)],
             '375b7795352c48febc969bff2357deff': [],
             '3862e0e76f9f4304a5c292c5622a4257': [('positive',
               0.8036268151801484)],
             '39844beedda4408fbfcfccbc2e674c55': [('positive',
               0.8036268151801484)],
             '3c7f92967c9f4dfa8e24f93a0c64e7a2': [('positive',
               0.8034925150648269)],
             '3d7e3d613bce4fa1a956ca51e18231e8': [],
             '3dd0ea8207ee471391a2b9e45758bf62': [('positive',
               0.999305913933766)],
             '3e68d07f718946948b048aa9e61035e1': [('positive',
               0.8036268151801484)],
             '3e6a9a8609de4c5aa15fa46e25840b50': [('positive',
               0.9870582038238152)],
             '3e7f3a815cce45cbaaadbdd80450445d': [('positive',
               0.8036268151801484)],
             '3f62e7773444447caddc35c9b586e9e4': [('positive',
               0.8036268151801484)],
             '3f6b986b194c4cec9fe136e849980290': [('positive',
               0.8036268151801484)],
             '3f89875821c34d8badf30dfc03719446': [('positive',
               0.8036268151801484)],
             '411bc55aa813494baf0bd469cda2f4b0': [('positive',
               0.8036268151801484)],
             '41a5025b7cff41a6ab157d91d1331021': [('positive',
               0.8036268151801484)],
             '4296b3e35a0d4efa9f9f973054c98b6e': [('positive',
               0.8036268151801484)],
             '440ec1cc944b498aba4942963885cc2d': [('positive',
               0.8036268151801484)],
             '45da2b606a0c4c0a8cc235f953213ae6': [('positive',
               0.8036268151801484)],
             '479441a3a04346de89e667d08b7f7ed6': [('positive',
               0.7146595030106319),
              ('positive', 0.7146595030106319),
              ('positive', 0.8036268151801484)],
             '4887902287f34d5c956a886e9545aa80': [('positive',
               0.8036268151801484)],
             '4ac854f0f46b4055b8ff0da4566df744': [('positive',
               0.8036268151801484)],
             '4df29934de2b48e1ba5859c098e292ba': [('positive',
               0.8036268151801484)],
             '4e6c4c7ad74e4fd0b12ad51debe3b5b8': [],
             '4ecf5da8e9b94e4fadac536083da2c4f': [('positive',
               0.8036268151801484)],
             '4ff9132d96da46809129e64d22a87849': [('positive',
               0.8036268151801484)],
             '4ffc88db5cda49d5a69aa9b8bff04692': [('positive',
               0.8036268151801484)],
             '4ffd5288540a4610afacda7d3d0a4648': [('positive',
               0.8036268151801484)],
             '507e1402884141008668cdcdac4c4f2b': [],
             '52074f66aeb8428da327ecf45622b3d4': [('positive',
               0.8034925150648269)],
             '5752b4cfe500452db7a637f86f1bc12f': [('positive',
               0.8036268151801484)],
             '579d63a3146b48ab81ab3b52ca978344': [('positive',
               0.8036268151801484)],
             '5a272f04bf3040ad9c9e6aa0f98b4f6f': [],
             '5c5617a9331845f6ae4cf4922ad95aba': [('positive',
               0.8036268151801484)],
             '5c59f73fae1048ef8c118e65e0e31745': [('positive',
               0.8036268151801484)],
             '5d3f09a84fe1434cbe7c1e60acd4e579': [('positive',
               0.8036268151801484)],
             '5e6e64d4b78b4da5bb3c18b01e2ee1de': [],
             '6142e661a437476a8190192f5806902c': [('positive',
               0.8036268151801484)],
             '61e9d3a219b849a3afc560b749d789c0': [('positive',
               0.8036268151801484)],
             '6217d9d755ec4d7db3ede4f658d30b81': [('positive',
               0.8036268151801484)],
             '62db30211bb84f3c9ec37d836bee1e4f': [],
             '6416fa5fab79440e8b761c4a96521b11': [('positive',
               0.8036268151801484)],
             '669a055ed49f4dd5a9d98a13b3e7facd': [('positive',
               0.8036268151801484)],
             '6bf241c1a83846b49aebaf485f6cb1e2': [('positive',
               0.8036268151801484)],
             '6c515f307c5945f48c9f03896766a653': [('positive',
               0.8036268151801484)],
             '6d2dd9fca4c840b5ac29f568b53b6341': [],
             '6e3d53d92d9f404a84c76d400b38a79d': [],
             '6edcc48a11ae41d6a6edadd2b2d6b136': [('positive',
               0.8036268151801484)],
             '6f3ee5224a4841adae317b5629bdeace': [('positive',
               0.8036268151801484)],
             '712ecad76fd04d78a6f62bd206152579': [('positive',
               0.8036268151801484)],
             '7289e897f5b0482d815d7b5eac35ced1': [],
             '72ab5aae3eae406ca40ce9da5ce3be95': [],
             '77631fe065fe467db71ae14a26ec7bde': [('positive',
               0.8036268151801484)],
             '7929158ea3674c31a9fa6cc1763a0d6f': [],
             '79664252303f406bbeb8eca76a528e83': [('positive',
               0.8036268151801484)],
             '7b3397b77d9e415c95986ea4a874ca8a': [('positive',
               0.8036268151801484)],
             '7b47469db42647ecb7f3ba222347690b': [],
             '7d98b2ec7e2c402c997c36a03c656748': [('positive',
               0.8036268151801484)],
             '7e2c0715bc184bec86f2ff6daafee232': [('positive',
               0.8036268151801484)],
             '7e5594f4998e41d1b56bf6a4c3429107': [],
             '7fe43d61605b4b269369f364d186c6d7': [],
             '809c44ef4178433489b9e9096d82d029': [],
             '834e421b227b48108f9f5a71ac1d7721': [('positive',
               0.8036268151801484)],
             '8351b7fe3f98466b8fda32fd45ce9cea': [],
             '83fb27a13c4c412581acc6f348a25be5': [('positive',
               0.8036268151801484)],
             '87f30e59bacb4d0c914364dd75e8e7ee': [('positive',
               0.8036268151801484)],
             '88cf9a10cc7a4b0e9c688ebc0ed712cb': [('positive',
               0.8036268151801484)],
             '88ec704f057d4466b7add04e1627d199': [],
             '88f6e6eb1b304c77846c3d9b55c78cfc': [('positive',
               0.8036268151801484)],
             '8a2e1de74c88486ab1a42bbb6c2876ee': [('positive',
               0.8036268151801484)],
             '8b010fcf811a404ca640aad3bba3ee3f': [('positive',
               0.8036268151801484)],
             '8b89e954249d45f5950e39391dc368f4': [('positive',
               0.8036268151801484)],
             '8e82853f73bb481cba53ee9cea167a4b': [],
             '8f4275eccfe940d2a0c89e0b5459dffb': [('positive',
               0.8036268151801484)],
             '9403a8f837c14ca5b1578a2bfb9b2a68': [('positive',
               0.8036268151801484)],}
    # evaluate(test_dict)
