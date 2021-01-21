# R-CNN에 대해서

## Roi(Region of Interest)

기존에는 이미지에서 object를 찾기 위해서 sliding window 방법을 이용했다.

![sliding Window](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwYMa4%2FbtqA6pruEvn%2FJJGkGhvMK2yIw1pVzKNGtk%2Fimg.png)

다양한 스케일로 모든 영역을 탐색한다. 이렇게 하는 것은 많이 비효율적이다. 그래서 이 '물체가 있을만한' 영역을 빠르게 찾아내는 알고리즘이 많들어 졌고, R-CNN 에서는 Selective search 알고리즘을 사용한다.

![Selective Search](https://mblogthumb-phinf.pstatic.net/MjAxNzAxMjRfMTQg/MDAxNDg1MjE5Mzk5NzI1.Jt_x39NqH2TeqKploHtfTH79scWdJgFXV4zHRV2NvfQg.D9DOiADp4yM1XGyzk3Kkx6MuAjfqUu2ekTRerzc9nsMg.PNG.laonple/selective.png?type=w2)

간단하게 Selctive Saerch에 대해서 이야기하면 색감이나 재질 등 여러가지 기준을 바탕으로 작은 부분들로 쪼개고 그 다음에 합치고 합치면서 물체가 있을만한 박스들을 찾는다. Region proposal이라고 부른다.

----
## R-CNN

R-CNN의 계열에는 R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN이 있다.

Fast R-CNN은 R-CNN의 단점을 보완하기 위해서 만들어졌고, Faster R-CNN은 Fast R-CNN의 한계를 보완하기 위해서 만들어졌다. Mask R-CNN은 Faster R-CNN에서 각각의 box에 mask를 씌워주는 모델이다.

R-CNN에 대해서 먼저 알아보자

1. Image를 입력받음
2. Selective Search를 통해서 region proposal을 만들고 동일한 사이즈로 만들기 위해서 image warping을 한다.
3. 각각의 warped image를 CNN 모델에 넣는다.
4. CNN을 통해서 나온 feature map을 활용해서 SVM을 통한 Classificatin과 regressor을 통한 Bounding box regression을 한다.

이런 프로세스로 돌아가게 된다. 그림으로 보기 좋게 나타내면 이렇게 된다.

![R-CNN 구조](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdmFi2%2FbtqAQ38E2v3%2FJMXznsWZsX3YQAuTkKtpWK%2Fimg.png)

이런 구조에는 단점들이 있다. 첫번째는 Selective Search를 통해 얻어내 region proposal 만큼 CNN을 통과해야 한다. 그리고 두번째로 큰 저장공간을 요구하고 많이 느렸다. 그리고 학습이 세단계로 진행이 된다.

---
## Fast R-CNN

그래서 R-CNN의 단점을 보완하기 위해서 Fast R-CNN이 만들어 졌다.

![Fast R-CNN 구조](https://i.imgur.com/G0hwkMF.png)

입력으로 전체 이미지와 object proposal을 사용한다. 전체 이미지를 이용해서 네트워크 과정으로 Conv feature map을 생성한다. 그리고 각 RoI에 대해서 Conv feature map을 통해서 RoI feature vector를 생성한다. 그 다음에 FC 층을 지나면서 각 RoI에 대해서 softmax 확률값과 class 별 Bounding box regression offset을 출력한다. 

Fast R-CNN에서는 기존의 R-CNN에서 사용하는 region-wise sampling 대신에 hierarchical sampling을 사용해 N개의 이미지를 미리 뽑고 그중에서 R 개의 RoI를 사용해 학습한다. 그리고 최종 classifier와 regression을 하나의 single state로 만들어서 fine-tunning 이 가능하게 되었다. 하나의 스테이지로 합쳐서 더 편리해졌다.

그런 방법을 이용해서 속도를 조금 더 개선했다.

![Fast R-CNN vs R-CNN](https://i.imgur.com/gYKd37p.png)

----
## Faster R-CNN

Fast R-CNN의 한계를 보완하기 위해서 나왔다. Fast R-CNN 에서 속도를 느리게 하는 주된 원인인 region proposal을 구하는 것이다. Faster R-CNN 에서는 딥러닝 구조를 이용했다. 기존의 Selective Search를 쓰지 않고 Region Proposal Network(RPN)을 통해서 RoI를 계산하게 된다. 그래서 GPU를 통한 RoI 연산이 가능해졌고 정확도도 많이 향상이 되었다.

Faster R-CNN은 두 개의 모듈로 구성된다.

1. Region proposal을 하는 Deep Conv Network
2. 제안된 영역을 사용하는 Fast R-CNN

![Faster R-CNN 구조](https://www.researchgate.net/profile/Zhipeng_Deng/publication/324903264/figure/fig2/AS:640145124499471@1529633899620/The-architecture-of-Faster-R-CNN.png)

![Faster R-CNN 구조_2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbUjRYz%2FbtqAWb0p8cv%2Fdx8Ky33sdZtb2RKQ8sQxZK%2Fimg.png)

맨처음 Feature map을 먼저 추출한다음 RPN에 전해주고 RoI를 구한다. 여기서 구한 RoI와 feature map으로 Fast R-CNN을 돌리게 된다.

RPN은 이런 구조로 되어 있다.

![RPN](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb7xNNb%2FbtqAYHyrFDU%2FJDkko5dBYTMzZV96AcpakK%2Fimg.png)

<details>
<summary> RPN 세부 동작 원리 </summary>
<div markdown="1">

1. CNN을 통해 뽑아낸 피쳐 맵을 입력으로 받습니다. 이 때, 피쳐맵의 크기를 H x W x C로 잡습니다. 각각 가로, 세로, 체널 수 입니다.

2. 피쳐맵에 3x3 컨볼루션을 256 혹은 512 체널만큼 수행합니다. 위 그림에서 intermediate layer에 해당합니다. 이 때, padding을 1로 설정해주어 H x W가 보존될 수 있도록 해줍니다. intermediate layer 수행 결과 H x W x 256 or H x W x 512 크기의 두 번째 피쳐 맵을 얻습니다.

3. 두 번째 피쳐맵을 입력 받아서 classification과 bounding box regression 예측 값을 계산해주어야 합니다. 이 때 주의해야할 점은 Fully Connected Layer가 아니라 1 x 1 컨볼루션을 이용하여 계산하는 Fully Convolution Network의 특징을 갖습니다. 이는 입력 이미지의 크기에 상관없이 동작할 수 있도록 하기 위함이다.

4. 먼저 Classification을 수행하기 위해서 1 x 1 컨볼루션을 (2(오브젝트 인지 아닌지 나타내는 지표 수) x 9(앵커 개수)) 체널 수 만큼 수행해주며, 그 결과로 H x W x 18 크기의 피쳐맵을 얻습니다. H x W 상의 하나의 인덱스는 피쳐맵 상의 좌표를 의미하고, 그 아래 18개의 체널은 각각 해당 좌표를 앵커로 삼아 k개의 앵커 박스들이 object인지 아닌지에 대한 예측 값을 담고 있습니다. 즉, 한번의 1x1 컨볼루션으로 H x W 개의 앵커 좌표들에 대한 예측을 모두 수행한 것입니다. 이제 이 값들을 적절히 reshape 해준 다음 Softmax를 적용하여 해당 앵커가 오브젝트일 확률 값을 얻습니다.

5. 두 번째로 Bounding Box Regression 예측 값을 얻기 위한 1 x 1 컨볼루션을 (4 x 9) 체널 수 만큼 수행합니다. 리그레션이기 때문에 결과로 얻은 값을 그대로 사용합니다.

6. 이제 앞서 얻은 값들로 RoI를 계산해야합니다. 먼저 Classification을 통해서 얻은 물체일 확률 값들을 정렬한 다음, 높은 순으로 K개의 앵커만 추려냅니다. 그 다음 K개의 앵커들에 각각 Bounding box regression을 적용해줍니다. 그 다음 Non-Maximum-Suppression을 적용하여 RoI을 구해줍니다.

</div>
</details>

Faster R-CNN은 training 시키기 위해 4단계 걸쳐서 training 하는 aAlternating Training 기법을 이용한다.

1. RPN은 Pretrain model인 ImageNet으로 초기화 해서 region proposal task를 위해 end to end로 학습한다.
2. 학습된 RPN을 사용해서 Fast R-CNN의 모델을 학습한다.
3. RPN을 학습하는 데 공통된 Conv layer는 고정하고 RPN에 연결된 층만 학습한다.
4. 공유된 Conv layer를 고정시키고 Fast R-CNN의 학습을 진행한다.

![Faster R-CNN training](https://i.imgur.com/xYCyHKY.png)

----
## Mask R-CNN

Faster R-CNN까지 전부 다 object Dection을 목표로 했다. Mask R-CNN은 더 발전해서 Instance Segmentation을 적용하는 모델이다. classify를 하는 것뿐만 아니라 identify까지 할 수 있게 한다.

> ## Image Segmentation에 대해서
>
> Image segmentation은 이미지의 각 영역을 분할해서 각 object에 맞게 합쳐주는 것을 말한다.
>
> ![Image segmentation](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpaJ7s%2FbtqB2vpB4BG%2FTu057OIbPZdBJwK3IfPGK0%2Fimg.png)
>
> Image segmentation의 대표적인 방법으로 Semantic segmentation과 Instance segmentation이 있다. Semantic 같은 class인 object들은 같은 영역 혹은 색으로 분할한다. 반대로 Instance Segmentation은 같은 class 여도 다른 Instance로 구분해준다. 

그래서  Instance segmentation을 적용하기 위해서는 objection detection과 semantic segmenation을 동시해 해줘야 한다. 그래서 각 RoI에 mask segmentation을 해주는 FCN을 추가해줬다. 그리고 기존의 RoI Pooling 방법은 위치 정보가 별로 중요하지 않아서 원본 위치 정보가 왜곡이 되었다. Classification task에는 문제가 생기지 않지만  Segmentation task에서는 문제가 생긴다. 그래서 RoI Pooling 대신에 RoI Align을 사용한다.

그리고 mask prediction과 class prediction을 decouple(분리)해서 다른 클래스 고려할 필요없이 binary mask를 predict하면 되기 때문에 성능이 많이 향상되었다.

![Mask R-CNN](https://t1.daumcdn.net/cfile/tistory/9906504F5BD9338E0D)