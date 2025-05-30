import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# From the old codebase
results = {"Logistic Regression": [0.27994235165490505, 0.28533445979524896, 0.2870241526687208, 0.28595567041049597, 0.2895089951297088, 0.28846536129609385, 0.2844647649339032, 0.28660172945035284, 0.288018089653116, 0.28463870390617235], "KNN": [0.24269456316469537, 0.2449806182288043, 0.2492545472617036, 0.24803697445581951, 0.2459497067885896, 0.2479127323327701, 0.24701818904681444, 0.24199880727561873, 0.24413577179206838, 0.2448812245303648], "Random Forest": [0.2717175231090349, 0.27350660968094626, 0.2775569028923566, 0.27740781234469736, 0.27484842460987974, 0.27827750720604316, 0.271692674684425, 0.27295994433952886, 0.27397872974853393, 0.27566842262200575], "Cell Type": [0.20644071165888084, 0.20840373720306132, 0.21387039061723487, 0.2120067587714939, 0.2100437332273134, 0.21317463472815823, 0.21180797137461485, 0.2049001093330683, 0.2069376801510784, 0.21155948712851605], "Cell Type KNN": [0.2583490706689196, 0.2673442003776961, 0.26600238544876253, 0.26612662757181194, 0.26868601530662956, 0.26806480469138255, 0.26525693271046613, 0.26220057648345096, 0.2626229997018189, 0.2647599642182686], "Geneformer_L{l}": 0.31097701029420977}

models = ['Cell Type Model', 'k-Neighbors', 'Cell Type k-Neighbors', 'Random Forest',
    'Logistic Regression', 'Murine Geneformer']

sns.set_style("ticks")
plt.figure(figsize=(5, 7))
ordering = np.argsort([np.mean(m) for m in results.values()])
plt.bar(models,
        np.array([np.mean(m) for m in results.values()])[ordering]*100,
        yerr=np.array([np.std(m) for m in results.values()])[ordering]*100, facecolor='k', capsize=5)
# plt.hlines(100/42., -.5,5.5,'r','--')
# plt.xlabel("Method", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(rotation=75, fontsize=12,  ha='right')
plt.yticks(fontsize=12)
plt.title("Single Cell Area Classification", fontsize=14)
plt.tight_layout()
plt.savefig("single_cell_area_classification_comparison.pdf", dpi=300)  # Save the plot as pdf