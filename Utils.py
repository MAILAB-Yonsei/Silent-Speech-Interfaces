import numpy as np
from sklearn.utils import shuffle
import os

seed = 222
np.random.seed(seed)


# alphabet_list = ['alpha','bravo', 'charlie','delta','echo','foxtrot','golf','hotel','india','juliet','kilo', 'lima','mike','november','oscar','papa',
#                  'quebac','romeo','sierra','tango','uniform','victor','whiskey','xray','yankee','zulu']

al = ['ABSOLUTELY', 'ACCUSED', 'AFTERNOON', 'AGREEMENT', 'ALLEGATIONS', 'ALMOST', 'AREAS', 'AUTHORITIES', 'BECOME', 'BEFORE', 'BEHIND', 'BELIEVE', 'BENEFIT', 'BETWEEN', 'CAMERON', 'CAMPAIGN', 'CHIEF', 'COMMUNITY', 'CONFLICT', 'CRIME', 'CUSTOMERS', 'DEGREES', 'DESCRIBED', 'DESPITE', 'DETAILS', 'ECONOMIC', 'EMERGENCY', 'ENGLAND', 'EUROPE', 'EUROPEAN', 'EVERYBODY', 'FAMILIES', 'FAMILY', 'FOLLOWING', 'FORMER', 'GERMANY', 'GLOBAL', 'HOMES', 'HOSPITAL', 'HUNDREDS', 'INCREASE', 'INFORMATION', 'INQUIRY', 'INVESTMENT', 'IRELAND', 'ISLAMIC', 'ITSELF', 'LEADERSHIP', 'LEAVE', 'MAJORITY', 'MEMBERS', 'MIGRANTS', 'MOMENT', 'MORNING', 'MOVING', 'NUMBERS', 'OBAMA', 'OFFICERS', 'OFFICIALS', 'OPERATION', 'OPPOSITION', 'PARLIAMENT', 'PARTS', 'PATIENTS', 'PEOPLE', 'PERHAPS', 'POLICY', 'POLITICIANS', 'POSSIBLE', 'POTENTIAL', 'PRIME', 'PRIVATE', 'PROBLEMS', 'PROCESS', 'PROVIDE', 'QUESTIONS', 'RECORD', 'REFERENDUM', 'REMEMBER', 'REPORTS', 'RESPONSE', 'SCOTLAND', 'SECRETARY', 'SIGNIFICANT', 'SIMPLY', 'SMALL', 'SUNSHINE', 'TEMPERATURES', 'THEMSELVES', 'THOUSANDS', 'TOMORROW', 'VICTIMS', 'WEAPONS', 'WEEKEND', 'WELCOME', 'WELFARE', 'WESTERN', 'WESTMINSTER', 'WITHOUT', 'WOMEN']

# dataset_alphabets = [100 for alphabet in al]

dataset_alphabets =[]
for i in range(100):  ##class num
    dataset_alphabets.append(8)   ## number of sample of each class.

# def normalize(x):
#     x = np.array(x)
#     min_ = x.min()
#     max_ = x.max()
#     # x = (x - min_)/ (max_ - min_)
#     x = x - min_
#     x /= max_
#     return x

def normalize(data):
    # data = (data - np.min(data))(np.max(data)-np.min(data))
    data = data - np.min(data)
    data = data/np.max(data)
    return data

def split_process(X,Y):
    X1_data, Y1_data = shuffle(X, Y)
    train_X1 = X1_data[0:int(len(X1_data) * 0.85), :]
    train_Y1 = Y1_data[0:int(len(X1_data) * 0.85)]
    test_X1 = X1_data[int(len(X1_data) * 0.85):, :]
    test_Y1 = Y1_data[int(len(X1_data) * 0.85):]

    return train_X1, train_Y1, test_X1, test_Y1

def divide_dataset(X,Y):

    train_X = np.array([])
    train_Y = np.array([])
    test_X   = np.array([])
    test_Y  = np.array([])
    for i in range(len(dataset_alphabets)):
        start = sum(dataset_alphabets[:i])
        stop = sum(dataset_alphabets[:i+1])

        train_X_tmp, train_Y_tmp, test_X_tmp, test_Y_tmp = split_process(X[start:stop],Y[start:stop])

        if i == 0:
            train_X = train_X_tmp
            train_Y = train_Y_tmp
            test_X  = test_X_tmp
            test_Y  = test_Y_tmp

        else:
            train_X = np.concatenate((train_X,train_X_tmp),axis=0)
            train_Y = np.concatenate((train_Y, train_Y_tmp), axis=0)
            test_X = np.concatenate((test_X, test_X_tmp), axis=0)
            test_Y = np.concatenate((test_Y, test_Y_tmp), axis=0)


    train_X, train_Y = shuffle(train_X, train_Y)
    return train_X, train_Y, test_X, test_Y


def shuffle_each_class(X,Y):
    X1_data, Y1_data = shuffle(X, Y)

    return X1_data, Y1_data

def shuffled_data(X,Y):

    train_X = np.array([])
    train_Y = np.array([])

    for i in range(len(dataset_alphabets)):
        start = sum(dataset_alphabets[:i])
        stop = sum(dataset_alphabets[:i+1])



        train_X_tmp, train_Y_tmp = shuffle_each_class(X[start:stop],Y[start:stop])



        if i == 0:
            train_X = train_X_tmp
            train_Y = train_Y_tmp


        else:
            train_X = np.concatenate((train_X, train_X_tmp), axis=0)
            train_Y = np.concatenate((train_Y, train_Y_tmp), axis=0)


    return train_X, train_Y

def crossval_split_process(X,Y, k):
    # X1_data, Y1_data = shuffle(X, Y)
    X1_data, Y1_data = X, Y

    test_s = int(len(X1_data) * 0.2)*k
    test_e = int(len(X1_data) * 0.2)*(k+1)
    print('start: {} end: {}'.format(test_s,test_e))
    test_X1 = X1_data[test_s: test_e]
    test_Y1 = Y1_data[test_s: test_e]

    train_X1 = [X1_data[n] for n in range(len(X1_data)) if n not in range(test_s, test_e)]
    train_Y1 = [Y1_data[p] for p in range(len(X1_data)) if p not in range(test_s, test_e)]

    train_X1 = np.array(train_X1)
    train_Y1 = np.array(train_Y1)
    test_X1 = np.array(test_X1)
    test_Y1 = np.array(test_Y1)

    return train_X1, train_Y1, test_X1, test_Y1

def crossval_divide_dataset(X,Y,k):


    train_X = np.array([])
    train_Y = np.array([])
    test_X = np.array([])
    test_Y = np.array([])
    for i in range(len(dataset_alphabets)):
        start = sum(dataset_alphabets[:i])
        stop = sum(dataset_alphabets[:i + 1])

        train_X_tmp, train_Y_tmp, test_X_tmp, test_Y_tmp = crossval_split_process(X[start:stop], Y[start:stop], k)

        if i == 0:
            train_X = train_X_tmp
            train_Y = train_Y_tmp
            test_X =   test_X_tmp
            test_Y =   test_Y_tmp

        else:
            train_X = np.concatenate((train_X, train_X_tmp), axis=0)
            train_Y = np.concatenate((train_Y, train_Y_tmp), axis=0)
            test_X = np.concatenate((test_X, test_X_tmp), axis=0)
            test_Y = np.concatenate((test_Y, test_Y_tmp), axis=0)

        print(train_Y_tmp)
        print(test_Y_tmp)


    train_X, train_Y = shuffle(train_X, train_Y)


    return train_X, train_Y, test_X, test_Y


def tsne_plot(features, labels, prefer_class_num, prefer_class_namelist, save_name, tsne_label_plot=True):
    import matplotlib
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=10.0, n_iter=2000, learning_rate=100)    
    
    fornum = 0
    for i in prefer_class_num:
        idx = np.where(labels==i)[0]

        temp = features[idx]
        temp_label = labels[idx]

        if fornum ==0:
            X = temp
            Y = temp_label
            fornum += 1
        else:
            X = np.append(X, temp, axis=0)
            Y = np.append(Y, temp_label, axis=0)
            fornum += 1

    X_2d = tsne.fit_transform(X)
    
    # for i in range(10):
    #     temp = (i+1)*np.ones((20, 2)) + np.random.rand(20, 2)
    #     X_2d[20*i:20*(i+1)] = temp

    plt.figure(figsize=(16, 16))
    
    def get_cmap(n, name='rainbow'):
        custom_cmap = plt.cm.get_cmap(name, n)
        cmap_list = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'rosybrown', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'salmon', 'tomato', 'darksalmon', 'coral', 'orangered', 'lightsalmon', 'sienna', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru','darkorange', 'burlywood', 'tan',  'blanchedalmond', 'moccasin', 'orange', 'wheat',  'darkgoldenrod', 'goldenrod', 'gold', 'khaki', 'palegoldenrod', 'darkkhaki', 'olive', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'chartreuse', 'lawngreen',  'darkseagreen', 'palegreen', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'springgreen', 'mediumspringgreen', 'mediumaquamarine', 'aquamarine', 'turquoise', 'lightseagreen', 'mediumturquoise',  'paleturquoise', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'aqua', 'cyan', 'darkturquoise', 'cadetblue', 'powderblue',  'deepskyblue', 'skyblue', 'steelblue', 'dodgerblue', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey', 'cornflowerblue', 'royalblue', 'lavender', 'midnightblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'thistle', 'plum', 'violet', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'lavenderblush', 'palevioletred', 'crimson', 'pink', 'lightpink']       
        cmap = matplotlib.colors.ListedColormap(cmap_list)
        # return custom_cmap
        return cmap

    cmap = get_cmap(len(prefer_class_namelist))
    
    fornum=0
    
    for i, label in zip(prefer_class_num, prefer_class_namelist):
        if tsne_label_plot:
            plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=cmap(fornum), label=label)
        else:
            plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=cmap(fornum))
        
        # plt.legend()
        # plt.rc('legend', fontsize=5.3)
        fornum += 1
    
    plt.savefig('./tsne_' + save_name + '.png')




def confusion_matrix_plot(confusion_matrix, label_name, save_name, cmap='Blues'):
    # Visualize confusion matrix
    import matplotlib
    import matplotlib.pyplot as plt
    from itertools import product
    
    def get_cmap(n, name='Reds'):
        custom_cmap = plt.cm.get_cmap(name, n)
        cmap_list = ['white']
        for i in range(n-1):
            cmap_list.append(custom_cmap(i))
        
        cmap = matplotlib.colors.ListedColormap(cmap_list)
        # return custom_cmap
        return cmap

    cmap = get_cmap(20, cmap)
    

    fig, ax = plt.subplots()
    cm = confusion_matrix
    ln = label_name
    n_classes = cm.shape[0]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
    im_fig = ax.imshow(cm, interpolation='nearest', cmap=cmap, norm=norm)
    
    
    # cmap_min, cmap_max = im_fig.cmap(0), im_fig.cmap(256)

    # text_t = np.empty_like(cm, dtype=object)
    # # print text with appropriate color depending on background
    # thresh = (cm.max() + cm.min()) / 2.0

    # for i, j in product(range(n_classes), range(n_classes)):
    #     color = cmap_max if cm[i, j] < thresh else cmap_min

    #     text_cm = format(cm[i, j], '.2g')
    #     if cm.dtype.kind != 'f':
    #         text_d = format(cm[i, j], 'd')
    #         if len(text_d) < len(text_cm):
    #             text_cm = text_d

    #     text_t[i, j] = ax.text(j, i, text_cm, ha="center", va="center", color=color)
    
    # bounds=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    fig.colorbar(im_fig, ax=ax)
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    
    
    fig.set_size_inches(11, 10)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=ln,
           yticklabels=ln, )
    
    ax.set_ylim((n_classes - 0.5, -0.5))
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    plt.show()
    fig.savefig(save_name)