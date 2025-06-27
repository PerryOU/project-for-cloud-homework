# === Python代码文件: trainer.py (完整修改版) ===

import torch
from torch import nn
import os
import json
import torch.optim as optim
from tqdm import trange, tqdm
from sklearn.metrics import classification_report
from numpy import mean
from tabulate import tabulate
from loss import Bias_loss
import matplotlib.pyplot as plt  # 新增导入

torch.autograd.set_detect_anomaly(True)


def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


class Trainer:
    def __init__(self,
                 args,
                 DetectionModel,
                 tr_set,
                 tr_size,
                 dev_set,
                 dev_size):
        self.args = args
        self.tr_set = tr_set
        self.tr_size = tr_size
        self.dev_set = dev_set
        self.dev_size = dev_size
        self.total_loss = nn.CrossEntropyLoss()
        self.classifier = DetectionModel
        self.bias_loss = Bias_loss()

    def train(self):
        NET_Classifier = optim.Adam(self.classifier.parameters(), lr=self.args.lr, weight_decay=1e-5)
        train_acc_values, train_loss_values, bias_list = [], [], []
        test_precision_values, test_recall_values, test_f1_values, test_acc_values = [], [], [], []
        epoch_pbar = trange(self.args.num_epoch, desc="Epoch")
        best_acc = 0

        for epoch in epoch_pbar:
            self.classifier.train()
            cls_loss, total_train_acc, certain_losses = [], [], []
            dct_acc, emo_acc, ambigious_acc = [], [], []

            with tqdm(self.tr_set, desc=f"Epoch {epoch}", leave=False) as pbar:
                for batch in pbar:
                    batch = batch.to(self.args.device)
                    y = batch.y

                    pred, y_embed, env_single, uncertain, certainloss, gcn_out, dct_out, emo_out = self.classifier(
                        batch, epoch + 1
                    )

                    _, label = torch.max(pred, dim=1)

                    total_loss_val = self.total_loss(pred, y)
                    bias, bias_index = torch.min(uncertain, dim=1, keepdim=True)
                    biasloss_val = self.bias_loss(bias, y_embed, gcn_out, emo_out, dct_out, bias_index)

                    correct = evaluation(label, y) / len(label)

                    _, ambigious_label = torch.max(env_single[0], dim=1)
                    _, dct_label = torch.max(env_single[1], dim=1)
                    _, emo_label = torch.max(env_single[2], dim=1)

                    ambigious_correct = evaluation(ambigious_label, y) / len(label)
                    dct_correct = evaluation(dct_label, y) / len(label)
                    emo_correct = evaluation(emo_label, y) / len(label)

                    class_loss_val = total_loss_val + self.args.alpha * certainloss + self.args.beta * biasloss_val

                    NET_Classifier.zero_grad()
                    class_loss_val.backward()
                    NET_Classifier.step()

                    cls_loss.append(class_loss_val.item())
                    certain_losses.append(certainloss.item())
                    dct_acc.append(dct_correct)
                    ambigious_acc.append(ambigious_correct)
                    emo_acc.append(emo_correct)
                    total_train_acc.append(correct)
                    bias_list.append(biasloss_val.item())
                    pbar.set_postfix(loss=class_loss_val.item())

            train_loss_info_json = {"epoch": epoch, "Class_loss": mean(cls_loss), "certain loss": mean(certain_losses)}
            train_acc_info_json = {"epoch": epoch, "train Acc": mean(total_train_acc), "gcn acc": mean(ambigious_acc),
                                   "text acc": mean(emo_acc), "image acc": mean(dct_acc), "bias": mean(bias_list)}
            train_acc_values.append(mean(total_train_acc))
            train_loss_values.append(mean(cls_loss))
            print(f"\n{'#' * 10} TRAIN LOSSES: {str(train_loss_info_json)} {'#' * 10}")
            print(f"{'#' * 10} TRAIN ACCURACY: {str(train_acc_info_json)} {'#' * 10}")

            with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as fout:
                fout.write(json.dumps(train_loss_info_json) + "\n")
                fout.write(json.dumps(train_acc_info_json) + "\n")

            self.classifier.eval()
            valid_acc, ans_list, preds = [], [], []

            with torch.no_grad():
                for batch in self.dev_set:
                    batch = batch.to(self.args.device)
                    y = batch.y
                    pred, _, _, _, _, _, _, _ = self.classifier(batch, epoch + 1)
                    _, label = torch.max(pred, 1)
                    correct = evaluation(label, y) / len(label)
                    valid_acc.append(correct)
                    ans_list.extend(y.cpu().numpy())
                    preds.extend(label.cpu().numpy())

            report = classification_report(ans_list, preds, digits=4, output_dict=True, zero_division=0)
            test_precision_values.append(float(report["macro avg"]["precision"]))
            test_recall_values.append(float(report["macro avg"]["recall"]))
            test_f1_values.append(float(report["macro avg"]["f1-score"]))
            valid_info_json = {"epoch": epoch, "valid_Acc": mean(valid_acc)}
            test_acc_values.append(mean(valid_acc))
            print(f"{'#' * 10} VALID: {str(valid_info_json)} {'#' * 10}")
            self.print_result_table_handler(train_loss_values, train_acc_values, test_acc_values, test_precision_values,
                                            test_recall_values, test_f1_values, report)

            with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as fout:
                fout.write(json.dumps(valid_info_json) + "\n")
                fout.write(classification_report(ans_list, preds, digits=4, zero_division=0))

            if mean(valid_acc) > best_acc:
                best_acc = mean(valid_acc)
                torch.save(self.classifier, f"{self.args.ckpt_dir}/{self.args.name}ckpt.classifier")
                print(f'Saving model with acc {mean(valid_acc):.3f}\n')

        # --- 新增代码块：在所有epoch结束后调用绘图函数 ---
        print("\n训练完成，正在生成可视化结果图...")
        self.plot_training_results(
            train_loss_values,
            train_acc_values,
            test_acc_values,
            test_precision_values,
            test_recall_values,
            test_f1_values
        )
        print("所有可视化图表已生成完毕。")
        # --- 新增代码块结束 ---

    def print_result_table_handler(self, loss_values, acc_values,
                                   test_acc_values,
                                   test_precision_values, test_recall_values,
                                   test_f1_values, report, print_type='tabel',
                                   table_type='pretty'):
        def trend(values_list):
            if len(values_list) == 1:
                diff_value = values_list[-1]
                return '↑ ({:+.6f})'.format(diff_value)
            else:
                diff_value = values_list[-1] - values_list[-2]
                if values_list[-1] > values_list[-2]:
                    return '↑ ({:+.6f})'.format(diff_value)
                elif values_list[-1] == values_list[-2]:
                    return '~'
                else:
                    return '↓ ({:+.6f})'.format(diff_value)

        if print_type == 'tabel':
            avg_table = [
                ["train loss", loss_values[-1], trend(loss_values)],
                ["train acc", acc_values[-1], trend(acc_values)],
                ["test acc", test_acc_values[-1], trend(test_acc_values)],
                ["test pre", test_precision_values[-1], trend(test_precision_values)],
                ['test rec', test_recall_values[-1], trend(test_recall_values)],
                ['test F1', test_f1_values[-1], trend(test_f1_values)]
            ]
            avg_header = ['metric', 'value', 'trend']
            print((tabulate(avg_table, avg_header, floatfmt=".6f", tablefmt=table_type)))

            class_table = []
            for c in sorted(report.keys()):
                if c.isdigit():
                    metrics = report[c]
                    class_table.append([c, metrics["precision"], metrics["recall"], metrics["f1-score"],
                                        f'{metrics["support"]}/{report["macro avg"]["support"]}'])
            class_header = ['class', 'precision', 'recall', 'f1', 'support']
            print((tabulate(class_table, class_header, floatfmt=".6f", tablefmt=table_type)))
        else:
            print(("Average train loss: {}".format(loss_values[-1])))
            print(("Average train acc: {}".format(acc_values[-1])))
            print(("Average test acc: {}".format(test_acc_values[-1])))
            print(report)

    # --- 新增的绘图方法 ---
    def plot_training_results(self, train_loss, train_acc, val_acc, val_precision, val_recall, val_f1):
        """
        在训练结束后绘制并保存训练/验证曲线图。
        """
        epochs = range(1, self.args.num_epoch + 1)
        output_dir = self.args.output_dir
        run_name = self.args.name

        # 1. 绘制训练损失曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
        plt.title(f'Training Loss vs. Epochs ({run_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{run_name}_train_loss_curve.png")
        plt.close()
        print(f"训练损失曲线图已保存到: {output_dir / f'{run_name}_train_loss_curve.png'}")

        # 2. 绘制训练与验证准确率曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_acc, 'b-o', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-s', label='Validation Accuracy')
        plt.title(f'Training & Validation Accuracy vs. Epochs ({run_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{run_name}_accuracy_curves.png")
        plt.close()
        print(f"准确率曲线图已保存到: {output_dir / f'{run_name}_accuracy_curves.png'}")

        # 3. 绘制验证集上的 Precision, Recall, F1-Score 曲线
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, val_precision, 'g-o', label='Validation Precision (Macro Avg)')
        plt.plot(epochs, val_recall, 'm-s', label='Validation Recall (Macro Avg)')
        plt.plot(epochs, val_f1, 'y-^', label='Validation F1-Score (Macro Avg)')
        plt.title(f'Validation Metrics vs. Epochs ({run_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{run_name}_validation_metrics_curve.png")
        plt.close()
        print(f"验证集评估指标曲线图已保存到: {output_dir / f'{run_name}_validation_metrics_curve.png'}")
    # --- 绘图方法结束 ---
