# === Python代码文件: tester.py (已修复) ===

import torch
import csv
import os
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.nn import functional as F
import pickle

class Tester:
    def __init__(self, args, DetectionModel, test_set, test_size):
        self.args = args
        self.classifier = DetectionModel
        self.test_set = test_set
        self.test_size = test_size

    def test(self):
        # --- FIX: 从正确的路径加载模型 ---
        model_path = self.args.ckpt_dir / f"{self.args.name}ckpt.classifier"
        if not model_path.exists():
            # 如果在ckpt_dir找不到，就尝试从test_path加载 (为了向后兼容main.py中的--test_path)
            if self.args.test_path and self.args.test_path.exists():
                 model_path = self.args.test_path
            else:
                print(f"Error: Model not found at {model_path}")
                print(f"Please check --ckpt_dir or provide a valid --test_path.")
                return

        print(f"Loading model from: {model_path}")
        self.classifier = torch.load(model_path, map_location=self.args.device)
        self.classifier.eval()

        all_labels, all_ids = [], []
        all_preds_prob, all_gcn_prob, all_emo_prob, all_dct_prob = [], [], [], []
        all_pred_labels, all_gcn_labels, all_emo_labels, all_dct_labels = [], [], [], []
        y_embed_dict, gcn_embed_dict = {}, {}

        with torch.no_grad():
            for batch in tqdm(self.test_set, desc="Testing"):
                batch = batch.to(self.args.device)
                y, ids = batch.y, batch.id

                pred, y_embed, env_single, _, _, gcn_out, _, _ = self.classifier(
                    batch, 72  # 在测试时，epoch/global_step可以是一个固定值
                )

                # --- 收集结果 ---
                for i, item_id in enumerate(ids):
                    y_embed_dict[item_id] = {"label": int(y[i]), "embedding": y_embed[i].cpu()}
                    gcn_embed_dict[item_id] = {"label": int(y[i]), "embedding": gcn_out[i].cpu()}

                _, final_labels = torch.max(pred, 1)
                _, gcn_l = torch.max(env_single[0], 1)
                _, dct_l = torch.max(env_single[1], 1)
                _, emo_l = torch.max(env_single[2], 1)

                all_labels.extend(y.cpu().numpy())
                all_ids.extend(ids)

                all_preds_prob.extend(F.softmax(pred, dim=-1)[:, 1].cpu().numpy())
                all_gcn_prob.extend(F.softmax(env_single[0], dim=-1)[:, 1].cpu().numpy())
                all_dct_prob.extend(F.softmax(env_single[1], dim=-1)[:, 1].cpu().numpy())
                all_emo_prob.extend(F.softmax(env_single[2], dim=-1)[:, 1].cpu().numpy())

                all_pred_labels.extend(final_labels.cpu().numpy())
                all_gcn_labels.extend(gcn_l.cpu().numpy())
                all_dct_labels.extend(dct_l.cpu().numpy())
                all_emo_labels.extend(emo_l.cpu().numpy())

        # --- 报告和保存结果 ---
        print("\n---  Test Report ---")
        print(classification_report(all_labels, all_pred_labels, digits=4, zero_division=0))

        pickle.dump(y_embed_dict, open(os.path.join(self.args.output_dir, "y_embed_test.pkl"), "wb"))
        pickle.dump(gcn_embed_dict, open(os.path.join(self.args.output_dir, "gcn_embed_test.pkl"), "wb"))

        # --- FIX: THIS IS THE CORRECTED LINE AND BLOCK ---
        with open(self.args.output_dir / f"{self.args.name}_report.txt", mode="w") as f:
            f.write("---  Test Report ---\n")
            f.write(classification_report(all_labels, all_pred_labels, digits=4, zero_division=0))
            f.write("\n\n--- GCN Branch Report ---\n")
            f.write(classification_report(all_labels, all_gcn_labels, digits=4, zero_division=0))
            f.write("\n\n--- Image (DCT) Branch Report ---\n")
            f.write(classification_report(all_labels, all_dct_labels, digits=4, zero_division=0))
            f.write("\n\n--- Text (Emo) Branch Report ---\n")
            f.write(classification_report(all_labels, all_emo_labels, digits=4, zero_division=0))

        with open(self.args.output_dir / f"{self.args.name}_results.csv", mode="w", newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'true_label', 'predicted_label', 'prediction_prob', "gcn_prob", "emo_prob", "dct_prob"])
            for i in range(len(all_ids)):
                writer.writerow([
                    all_ids[i], all_labels[i], all_pred_labels[i], all_preds_prob[i],
                    all_gcn_prob[i], all_emo_prob[i], all_dct_prob[i]
                ])

        print(f"\nDetailed results saved to {self.args.output_dir / f'{self.args.name}_results.csv'}")
        print(f"Classification reports saved to {self.args.output_dir / f'{self.args.name}_report.txt'}")
