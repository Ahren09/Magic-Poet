## About Magic Poet
"Magic poet" is a Tang poem generator based on RNN. It was implemented using PyTorch. The model was trained using CUDA

The dataset can be found at [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry), a collection of more than 25,000 Chinese Tang poems. Alternatively, you can utilize preprocessed data using tang.npz](http://pytorch-1252820389.cosbj.myqcloud.com/tang.npz)

## Environment setup
- Install [PyTorch](http://pytorch.org)

- Install visdom
```Bash
 python -m visdom.server
```
- Alternatively, you can run:
```Bash
nohup python -m visdom.server &
``` 
## Train
Magic Poet supports a command line API. You can run the command to 

```Bash
python main.py train --plot-every=150\
					 --batch-size=128\
                     --pickle-path='tang.npz'\
                     --lr=1e-3 \
                     --env='poetry3' \
                     --epoch=50
```

The commandine arguments are specified：
```Python
    data_path = 'data/' # Path to store the poems
    pickle_path= 'tang.npz' # Preprocessed binary dataset 
    author = None # Learn from specific poets
    constrain = None # Length limit
    category = 'poet.tang' # Alternatively you can train with 'poet.song'
    lr = 1e-3 
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20  
    batch_size = 128
    maxlen = 125 # Max length of generated poem
    plot_every = 20 # Visualize every 20 batches
    env='poetry' # visdom env
    max_gen_len = 200 # Max length of generated poem
    debug_file='/tmp/debugp'
    model_path=None # Path of pretrained model
    prefix_words = '细雨鱼儿出,微风燕子斜。' # Set the tone of the poem
    start_words='闲云潭影日悠悠' # Start words
    acrostic = False
    model_prefix = 'checkpoints/tang' # Path for storing the verse.

```
## Generate
You can download the pre-trained model here[tang_199.pth](http://pytorch-1252820389.cosbj.myqcloud.com/tang_199.pth) to generate verses

Generating acrostic poem：

```Bash
python  main.py gen  --model-path='checkpoints/tang_199.pth' \
       --pickle-path='tang.npz' \
       --start-words='深度学习' \
       --prefix-words='江流天地外，山色有无中。' \
       --acrostic=True\
       --nouse-gpu
深居不可见，浩荡心亦同。度年一何远，宛转三千雄。学立万里外，诸夫四十功。习习非吾仕，所贵在其功。
```

Generating poem with start words

```Bash
python2 main.py gen  --model-path='model.pth' 
					 --pickle-path='tang.npz' 
					 --start-words='江流天地外，' # 诗歌的开头
					 --prefix-words='郡邑浮前浦，波澜动远空。' 
江流天地外，风日水边东。稍稍愁蝴蝶，心摧苎范蓬。云飞随海远，心似汉阳培。按俗朝廷上，分军朔雁通。封疆朝照地，赐劒豫章中。畴昔分曹籍，高名翰墨场。翰林推国器，儒冠见忠贞。臯宙非无事，姦邪亦此中。渥仪非贵盛，儒实不由锋。几度沦亡阻，千年垒数重。宁知天地外，长恐海西东。邦测期戎逼，箫韶故国通。蜃楼瞻凤篆，云辂接旌幢。別有三山里，来随万里同。烟霞临海路，山色落云中。渥泽三千里，青山万古通。何言陪宴侣，复使
```

### Compatibility
train 
- [x] GPU  
- [] CPU  
- [] Python2
- [x] Python3

test: 

- [x] GPU
- [x] CPU
- [] Python2
- [x] Python3


## Examples

- Acrostic mode

- Start with "苟利国家生死以", a famous Chinese verse


!["苟利国家生死以"](https://github.com/Ahren09/Magic-Poet/blob/master/examples/%E8%8B%9F%E5%88%A9%E5%9B%BD%E5%AE%B6.png)


```Bash
 python3  main.py gen  --model-path='checkpoints/tang_199.pth' \
                                     --pickle-path='tang.npz' \
                                     --start-words="深度学习" \
                                     --prefix-words="江流天地外，山色有无中。" \
                                     --acrostic=True\
                                     --nouse-gpu
深井松杉下，前山云汉东。度山横北极，飞雪凌苍穹。学稼落羽化，潺湲浸天空。习习时更惬，俯视空林濛。
```

- Normal mode
- Start with "我爱学习", which stands for "I love Studying"
!["我爱学习"](https://github.com/Ahren09/Magic-Poet/blob/master/examples/%E6%88%91%E7%88%B1%E5%AD%A6%E4%B9%A02.png)

- Start with "深度学习", which stands for "Deep Learning"
```Bash
python2  main.py gen    --model-path='checkpoints/tang_199.pth' \
                        --pickle-path='tang.npz' \
                        --start-words="深度学习" \
                        --prefix-words="庄生晓梦迷蝴蝶，望帝春心托杜鹃。" \
                        --acrostic=False\
                        --nouse-gpu
深度学习书不怪，今朝月下不胜悲。芒砀月殿春光晓，宋玉堂中夜月升。玉徽美，玉股洁。心似镜，澈圆珠，金炉烟额红芙蕖。红缕金钿舞凤管，夜妆妆妓。歌中有女子孙子，嫁得新年花下埽。君不见金沟里，裴回春日丛。歌舞一声声断，一语中肠千万里。罗帐前传，娉婷花月春，一歌一曲声声。可怜眼，芙蓉露。妾心明，颜色暗相思，主人愁，万重金。红粉，冉冉，芙蓉帐前飞。鸳鸯鬬鸭，绣衣罗帐，鹦鹉抹。凰翠忽，菱管。音舞，行路，蹙罗金钿
```
