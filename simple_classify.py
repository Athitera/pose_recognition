from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import stochastic_gradient
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
import pandas as pa
import numpy as np
import pydotplus


def get_sample_from_csv():
    csv_file = pa.read_csv('D:\\Dataset\\test.csv')
    y = csv_file['label_index']
    X = csv_file.drop('label_index', axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test, csv_file


def standard(X_train, X_test, y_train, y_test):
    # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


def LR(X_train, X_test, y_train, y_test):
    # 使用LR
    lr = LogisticRegression(C=1000.0, random_state=0)  # 初始化模型
    lr.fit(X_train, y_train)
    lr_y_predict = lr.predict(X_test)  # 预测
    y_pre = lr.predict(X_test)
    print(y_pre)
    print("Accuracy of LR Classifier:%f" % lr.score(X_test, y_test))
    print(classification_report(y_test, lr_y_predict, target_names=['11', '1']))


# 随机梯度下降
def SGD(X_train, X_test, y_train, y_test):
    sgdc = stochastic_gradient.SGDClassifier(max_iter=5) #初始化分类器
    sgdc.fit(X_train, y_train)
    sgdc_y_predict = sgdc.predict(X_test)
    print("Accuracy of SGD Classifier:%f" % sgdc.score(X_test, y_test))
    print(classification_report(y_test, sgdc_y_predict, target_names=['11', '1']))


def svm(X_train, X_test, y_train, y_test, csv_file):
    lsvc = LinearSVC()
    lsvc.fit(X_train, y_train)
    y_predict = lsvc.predict(X_test)
    print("Accuracy of SVM Classifier:%f" % lsvc.score(X_test, y_test))
   # print(classification_report(y_test, y_predict, target_names=csv_file.target_names.astype(str)))


# 决策树 0.953486
def tree(X_train, X_test, y_train, y_test):
    clt = DecisionTreeClassifier()
    clt.fit(X_train, y_train)
    y = clt.predict(X_test)
    print("Accuracy of tree Classifier:%f" % clt.score(X_test, y_test))
    dot_data = export_graphviz(clt, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("D:\\Dataset\\tree.pdf")


# 随机森林 0.991572
def random_forest(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_y_predict = rfc.predict(X_test)
    print("Accuracy of random forest classifier:%f" % rfc.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, csv_file = get_sample_from_csv()
    X_train, X_test, y_train, y_test = standard(X_train, X_test, y_train, y_test)
    #LR(X_train, X_test, y_train, y_test)
    #SGD(X_train, X_test, y_train, y_test)
    svm(X_train, X_test, y_train, y_test, csv_file)
    #tree(X_train, X_test, y_train, y_test)
    # random_forest(X_train, X_test, y_train, y_test)
