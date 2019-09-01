# coding:utf8
import sys, os
import torch
from data import get_data
from model import Net
import torch.nn as nn
# from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb

from utils import Visualizer


class Config(object):
    data_path = './data/'
    pickle_path = 'tang.npz'
    author = None
    constrain = None
    category = 'poet.tang'
    lr = 1e-3
    use_gpu = False
    epoch = 1
    batch_size = 128
    max_length = 125
    plot_every = 20
    save_every = 100
    use_env = True # Use visdom or not?
    env = 'poetry'
    max_gen_len = 124 # Length of generated poem
    debug_file = '/tmp/debugp'
    model_path = None # "./checkpoints/tang.pth"

    # Input verses
    prefix_words = "亦可赛艇"
    start_words = "苟利国家生死以"
    acrostic = False
    model_prefix = "./checkpoints/"

opt = Config()

# generate()
# Given the first few words (start_words), generate a complete poem 
def generate(net, start_words, ix2word, word2ix, prefix_words=None):
    results = []
    input = torch.Tensor([word2ix['<START>']]).view(1,1).long()
    start_words_len = len(start_words)

    # Enable GPU
    if opt.use_gpu:
        input = input.cuda()

    hidden = None

    # If we input prefix_words for artistic conception
    if prefix_words:
        for word in prefix_words:
            output, hidden = net(input, hidden)
            input = input.data.new([word2ix[word]]).view(1,1) # TODO: new() and view(1,1)
        

    # Generate output and hidden state
    for i in range(opt.max_gen_len):
        output, hidden = net(input, hidden)
        word = None

        # If start_words not used up, input start_words
        if i < start_words_len:
            word = start_words[i]
            input = input.data.new([word2ix[word]]).view(1,1)
            

        else:
            top_index = output.data[0].topk(1)[1][0].item()
            word = ix2word[top_index]
            input = input.data.new([top_index]).view(1,1)
        
        if word != '<EOP>':
            results.append(word)

    return results
    
def generate_acrostic(net, start_words, ix2word, word2ix, prefix_words=None, start_words_2=None):
    results = []
    start_word_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1,1).long()

    # Enable GPU
    if opt.use_gpu:
        input = input.cuda()

    # index: indicates number of sentences already generated
    index = 0

    # Start word 2
    index2, start_word_len2 = None, None
    if start_words_2:
        index2, start_word_len2 = 0, len(start_words_2)

    # pre_word: Last word; used as input for generating next character
    pre_word = '<START>'
    hidden = None

    # If 
    if prefix_words:
        for word in prefix_words:
            output, hidden = net(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = net(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        
        if pre_word in {u'。', u'！', '<START>'}:
            # If all prefix words have been used, continue; else skip
            if index != start_word_len:
                w = start_words[index]
                index += 1
                input = input.data.new([word2ix[w]]).view(1,1)

        elif pre_word == u'，':
            if index2 != start_word_len2:
                w = start_words_2[index2]
                index2 += 1
                input = input.data.new([word2ix[w]]).view(1,1)
        else:
            input = input.data.new([word2ix[w]]).view(1, 1)

        results.append(w)
        pre_word = w
    return results

def train(**kwargs):
    # Set parameters with each key-value pair
    for key, value in kwargs.items():
        setattr(opt, key, value)

    data, word2ix, ix2word = get_data(opt)
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    vis = Visualizer(env=opt.env)

    # Initialize the Net
    net = Net(len(word2ix), 128, 256) # TODO: Why 128 and 256?
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    # Load pre-trained model
    if opt.model_path:
        net.load_state_dict(torch.load(opt.model_path))

    # TODO
    loss_meter = meter.AverageValueMeter()

    # Go over each epoch
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for step, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous()
            optimizer.zero_grad()

            input, target = data[:-1, :], data[1:, :]
            output, _ = net(input)

            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            # Visualize
            if (1 + step) % opt.plot_every == 0:
                print('Epoch: ', epoch, '| Step: ', step)
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])



                # 诗歌原文
                poets = [[ix2word[w] for w in data[:, _iii].tolist()] for _iii in range(data.shape[1])][:16]
                vis.text('</br>'.join([''.join(poet) for poet in poets]), win=u'origin_poem')
                
                gen_poets = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'苟利国家生死以'):
                    gen_poet = ''.join(generate_acrostic(net, word, ix2word, word2ix))
                    gen_poets.append(gen_poet)
                
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poets]), win=u'gen_poem')
            torch.save(net.state_dict(), '%s_%s.pth'% (opt.model_prefix, epoch))

def generate_pretrained(path, start_words, ix2word, word2ix, prefix_words=None):
    net = Net(len(word2ix), 128, 256)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    result = generate(net, start_words, ix2word, word2ix, prefix_words)
    sys.stdout.write("Topic: ")
    sys.stdout.write(prefix_words)
    print()
    print(''.join(result))

def generate_acrostic_pretrained(path, start_words, ix2word, word2ix, prefix_words=None, start_words_2=None):
    net = Net(len(word2ix), 128, 256)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    result = generate_acrostic(net, start_words, ix2word, word2ix, prefix_words, start_words_2)
    for r in result:
        sys.stdout.write(r)
        if r==u'。':
            print()
    sys.stdout.write(''.join(result))


if __name__ == '__main__':
    _, word2ix, ix2word = get_data(opt)
    # 苟利国家
    # generate_acrostic_pretrained('./checkpoints/tang_199.pth', u'苟利国家生死以', ix2word, word2ix, start_words_2=u'岂因祸福避趋之')

    # 一一二七事件有感
    # generate_pretrained('./checkpoints/tang_199.pth', u'在下知错', ix2word, word2ix, u"一一二七事件有感")
    
    # 我爱学习
    # generate_acrostic_pretrained('./checkpoints/tang_199.pth', u'我爱学习', ix2word, word2ix, start_words_2=u'学习爱我')

    '''
    # Generate with each char in list
    for word in list(u'苟利国家生死以'):
        generate_acrostic_pretrained('./checkpoints/tang_199.pth', word, ix2word, word2ix)
    '''
    
    train()

