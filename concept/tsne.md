

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2).fit_transform(train_x)

tsne_df = pd.DataFrame({'x': tsne[:, 0], 'y':tsne[:, 1], 'classes':train_y})

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x = 'x', y = 'y',
    hue = 'classes',
    palette = sns.color_palette("Set1", 10),
    data = tsne_df,
    legend = "full",
    alpha = 0.4
)

plt.title("tSNE")

plt.savefig('HW6/TSNE.png', bbox_inches='tight')  
plt.show()
```


data를 2차원 공간으로 축소하기 위해 'n_componets'를 2로 선언하였다. 
'fit_transform' 메소드를 이용하여 train_x 데이터를 임베딩된 공간에 맞추고, 변환된 결과를 사용한다. 
결과 시각화를 위해 각 컬럼을 DataFrame에 넣고, 데이터 샘플을 2차원 공간에 각 클래스별로 플로팅한다. 


### 📊 Experimental Result 
PCA는 데이터 샘플의 클래스에 관계없이 원본 데이터의 특성을 가장 잘 설명할 수 있는 축을 사용하여 차원을 줄인다. 따라서, 데이터의 특성은 알 수 있지만 각 데이터들의 클래스 간의 구분이 명확하지 않다. 반면, TSNE는 원본 데이터들의 거리를 잘 보존하는 2차원 표현을 찾기 위해 멀리 떨어진 원본 데이터는 더 멀게, 가까운 데이터는 더 가깝게 만든다. 따라서, 결과 사진에서 볼 수 있듯이 PCA보다 데이터들의 클래스를 더 잘 구분해준다. 
하지만, tsne는 pca에 비해 계산시간이 더 많이 소요된다. 데이터가 선형적이라는 전제 조건을 가질 때는 PCA를 사용하는 것이 더 효율적이라고 생각한다. 어떤 데이터를 사용하느냐와 목적에 따라 비용을 고려하여 차원 축소 방법을 택해야 한다고 생각된다.

