import json

pred_json = 'res/2022-05-27-19.json'
gt_json = "data/gt_datanode1200_querynode29_query1000_maxlabel10.json"

fp = open(gt_json, 'r')
f_gt = json.load(fp)

fp = open(pred_json, 'r')
f_pred = json.load(fp)
true = 0
sum=0
for j in range(len(f_gt)):
    gt = f_gt[j]
    pred = f_pred['B0S%d'%j]
    pred = {a:b for a,b in pred}
    gt = {int(a):b for a,b in gt.items()}
    for t in gt.keys():
        if t in pred.keys():
            true += (gt[t] == pred[t])
        else:
            print("key %d not in"%t, pred.keys())
        sum += 1
print(true, sum, true/sum)
    # print(pred.keys(), gt.keys())