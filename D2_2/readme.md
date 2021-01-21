# PA1

KNN 알고리즘을 구현하기 위해 먼저 데이터를 불러들어야 한다.

``` python
with gzip.open('Dataset\mnist.pkl.gz', 'rb') as f:
    try: 
        train_set, vaild_set, test_set = pickle.load(f, encoding='latin1')
    except:
        print("GZIP Error")
        exit()

train_image, train_label = train_set
val_image, val_label = vaild_set
test_image, test_label = test_set
```

먼저 pickle.load 명령어를 통해서 MNIST 데이터를 불러온다. train_set, vaild_set, test_set을 튜플 형태로 불러오고 후에 image와 label로 분리한다. image와 label은 np.ndarray의 형태로 있다.

``` python
def Image_show(n):
    image = test_image[n]
    label = test_label[n]
    image = image.reshape(28, 28)
    image.shape

    plt.figure(figsize=(4, 4))
    plt.title("answer: " + str(label))
    plt.imshow(image, cmap="gray")
    plt.show()
```

matplotlib 라이브러리를 이용해서 test_image와 test_label이 서로 맞는 값을 가리키고 있는 지 데이터를 잘 불러왔는지 확인한다. 코드를 실행하면 이렇게 된다.

![image_show](image\Image_show_run.PNG)

그 다음 거리를 구해야 하는데 거리를 구하는 방법으로 Euclidean Distance 방법을 이용했다.

![Euclidean Distance](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkGQX8%2Fbtqw7wGnkRe%2FxrbQXC5GVPlCVmJoR9Yz4K%2Fimg.png)

코드로 구현하면 

``` python
def U_Distance(test, train): #Euclidean Distance
    result = test - train
    result = np.square(result)
    result = result.sum()
    result = math.sqrt(result)
    return result
```

이렇게 된다. 그 다음에 클래스를 결정하는 함수를 작성하면 된다.

``` python
def DefineClass(test, k, batch_size):
    distance = []
    
    n_list = [0*i for i in range(10)]

    batch = np.random.randint(0, TRAIN_LEN, size = batch_size)

    for i in batch:
        distance.append([U_Distance(test, train_image[i]), train_label[i]])
    
    distance.sort()

    for i in range(k):
        n_list[distance[k][1]] +=1

    max_i = 0
    for i in range(10):
        if(n_list[i] >= n_list[max_i]):
            max_i = i

    return max_i
```

전체 train_set 갯수가 50000개가 있는데 그것 다 하나하나 거리를 구하면 연산 비용이 너무 많이 들기 때문에 batch라고 하나 설정해서 랜덤한 train_set을 하나 만들었다. 그 다음에 k 는 인접한 점의 갯수이다.

이렇게 해서 작성된 KNN 알고리즘의 하이퍼 파라미터는 거리를 결정하는 함수, k, batch_size 이렇게 3가지 이다.

~~하이퍼 파라미터를 변화해줘서 정확도가 어떻게 되는지 확인해보자.~~

확인해 보려고 했지만 연산 비용이 많이 들어서 cpu로는 작업이 많이 힘들었다. k 값의 변화에 따른 정확도 변화를 그래프로 나타내려고 했지만 너무 느렸다. 그래서 gpu로 작업하려고 CudaTool kit도 설치하고 pytorch도 설치하고 안 되서 CudaTool kit 다운그레이드도 해보고 했지만 프로그램이 작동될 때 외장그래픽을 쓰지 않는다. 그래서 하지 못했다. torch.cuda.is_available()이 True가 되고 get_device_name(0)하면 외장 그래픽 이름을 가르치고, tensor 만들어서 cuda 에 넣으면 cuda:0해서 들어가지만 gpu에서 돌아가지 않는다. 그래서 변화를 확인하지는 못했고, k가 5 이고 batch_size는 1000일때 정확도 74.42%가 나온다.

----
## Random Forest

Random Forest는 sklearn 라이브러리를 사용해서 구현한다.

```python

import numpy as np
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import gzip
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

LENGTH = 784
TRAIN_LEN = 50000
TEST_LEN = 10000

with gzip.open('Dataset\mnist.pkl.gz', 'rb') as f:
    try: 
        train_set, vaild_set, test_set = pickle.load(f, encoding='latin1')
    except:
        print("GZIP Error")
        exit()

train_image, train_label = train_set
val_image, val_label = vaild_set
test_image, test_label = test_set

rf = RandomForestClassifier(random_state = 0)
rf.fit(train_image, train_label)

pred = rf.predict(test_image)
print("Accuracy: ", accuracy_score(test_label, pred)*100, '%')

```

sklearn 을 이용해서 Random Forest를 구현했다. Random Forest는 라이브러리를 이용하면 구현이 간다하고 성능도 좋게 나오는 편이다. 여기서 하이퍼 파라미터를 상황에 적합하게 커스터마이징을 할 수 있다. 

1. max_depth
2. min_sample_split
3. max_leaf_nodes
4. max_samples_leaf
5. n_estimators
6. max_sample
7. max_features

7개의 파라미터를 적절하게 조절하여 성능을 변경할 수 있다. 
