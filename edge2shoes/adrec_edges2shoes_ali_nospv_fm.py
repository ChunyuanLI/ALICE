import os
import argparse
from datetime import datetime
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
from dataset import *
from model import *

import scipy
from progressbar import ETA, Bar, Percentage, ProgressBar

parser = argparse.ArgumentParser(description='PyTorch implementation of ALICE')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='edges2shoes', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=10000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='adrec_edges2shoes_ali_nospv_fm', help='choose among')
parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--gan_curriculum', type=int, default=10000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.9, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.9, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')
parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=500, help='Save test results every log_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every log_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    if args.task_name == 'edges2shoes':
        data,myidx = get_edge2photo_files(item='edges2shoes', test=False, use_spv_train=True, num_spv=0)
        [test_A, test_B] = get_edge2photo_files(item='edges2shoes', test=True)
    return data, myidx, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats[0:], fake_feats[0:]):
        #l2 = (real_feat - fake_feat) * (real_feat - fake_feat)
        #loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
        #loss = criterion( real_feat, fake_feat )
        #losses += loss
        loss = torch.mean((real_feat - fake_feat) * (real_feat - fake_feat))
        losses += loss

    return losses

def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
    labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1] ))
    labels_gen = Variable(torch.ones([dis_fake.size()[0], 1]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss

def get_ali_loss(dis_real, dis_fake, criterion, cuda):
    labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
    labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1] ))
    labels_gen_real = Variable(torch.zeros([dis_real.size()[0], 1]))
    labels_gen_fake = Variable(torch.ones([dis_fake.size()[0], 1]))
    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen_real = labels_gen_real.cuda()
        labels_gen_fake = labels_gen_fake.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_real, labels_gen_real ) * 0.5 + criterion( dis_fake, labels_gen_fake ) * 0.5

    return dis_loss, gen_loss



def main():

    global args
    args = parser.parse_args()


    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    task_name = args.task_name

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    result_path = os.path.join( args.result_path, args.task_name, args.model_arch )
    model_path = os.path.join( args.model_path, args.task_name, args.model_arch )

    data, myidx, test_style_A, test_style_B = get_data()
    # 84*11 x 3 x 64 x 64 , 85*11 x 3 x 64 x 64, 14*11 x 3 x 64 x 64, 14*11 x 3 x 64 x 64

    test = test_style_A + test_style_B


    if not args.task_name.startswith('car') and not args.task_name.endswith('car'):
        test_A = read_images( filenames=test, domain='A', image_size=args.image_size )
        test_B = read_images( filenames=test, domain='B', image_size=args.image_size )



    test_A = Variable( torch.FloatTensor( test_A ), volatile=True )
    test_B = Variable( torch.FloatTensor( test_B ), volatile=True )

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator_A = Generator(extra_layers=True)
    generator_B = Generator(extra_layers=True)
    discriminator_ali = ad_Discriminator()
    discriminator_ReconA = ad_Discriminator_fm1()
    discriminator_ReconB = ad_Discriminator_fm1()

    if cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_ali = discriminator_ali.cuda()
        discriminator_ReconA = discriminator_ReconA.cuda()
        discriminator_ReconB = discriminator_ReconB.cuda()



    data_size = len(data)
    n_batches = ( data_size // batch_size )

    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    #feat_criterion = nn.HingeEmbeddingLoss()
    feat_criterion = nn.MSELoss()
    spv_criterion = nn.MSELoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_ali.parameters(), discriminator_ReconA.parameters(), discriminator_ReconB.parameters())

    optim_gen = optim.Adam( gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    gen_loss_total = []
    dis_loss_total = []

    for epoch in range(epoch_size):

        _idx_A = list(range(len(data)))
        np.random.shuffle( _idx_A )
        _idx_B = list(range(len(data)))
        np.random.shuffle( _idx_B )
        data_A = np.array(data)[ np.array(_idx_A) ]
        data_B = np.array(data)[ np.array(_idx_B) ]



        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()


        for i in range(n_batches):


            pbar.update(i)

            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_ali.zero_grad()
            discriminator_ReconA.zero_grad()
            discriminator_ReconB.zero_grad()

            ############################################################################# un_spv
            _path_A = data_A[ i * batch_size: (i+1) * batch_size ]
            _path_B = data_B[ i * batch_size: (i+1) * batch_size ]
            A = read_images( filenames=_path_A, domain='A', image_size=args.image_size )
            B = read_images( filenames=_path_B, domain='B', image_size=args.image_size )
            A = Variable( torch.FloatTensor( A ) )
            B = Variable( torch.FloatTensor( B ) )

            if cuda:
                A = A.cuda()
                B = B.cuda()

            AB = generator_B(A)
            BA = generator_A(B)
            # Use discriminator to replace Reconstruction Loss
            ABA = generator_A(AB)
            BAB = generator_B(BA)
            A_t = torch.cat([A,A] ,1).cuda() # 64 x 9 x 64 x 64
            A_f = torch.cat([A,ABA] ,1).cuda() # 64 x 9 x 64 x 64
            ReconA_dis_real, ReconA_feats_real = discriminator_ReconA( A_t )
            ReconA_dis_fake, ReconA_feats_fake = discriminator_ReconA( A_f )

            dis_loss_ReconA, gen_loss_ReconA = get_gan_loss( ReconA_dis_real, ReconA_dis_fake, gan_criterion, cuda )
            fm_loss_ReconA = get_fm_loss(ReconA_feats_real, ReconA_feats_fake, feat_criterion)


            B_t = torch.cat([B,B] ,1).cuda() # 64 x 9 x 64 x 64
            B_f = torch.cat([B,BAB] ,1).cuda() # 64 x 9 x 64 x 64
            ReconB_dis_real, ReconB_feats_real = discriminator_ReconB( B_t )
            ReconB_dis_fake, ReconB_feats_fake = discriminator_ReconB( B_f )

            dis_loss_ReconB, gen_loss_ReconB = get_gan_loss( ReconB_dis_real, ReconB_dis_fake, gan_criterion, cuda )
            fm_loss_ReconB = get_fm_loss(ReconB_feats_real, ReconB_feats_fake, feat_criterion)


            # Real/Fake GAN Loss (A)
            tuple_1 = torch.cat([A,AB] ,1).cuda() # 64 x 6 x 64 x 64
            tuple_2 = torch.cat([BA,B] ,1).cuda() # 64 x 6 x 64 x 64
            dis_real, feats_real = discriminator_ali( tuple_1 )
            dis_fake, feats_fake = discriminator_ali( tuple_2 )

            dis_loss, gen_loss = get_ali_loss( dis_real, dis_fake, gan_criterion, cuda )
            fm_loss = get_fm_loss(feats_real, feats_fake, feat_criterion)
            if iters < args.gan_curriculum:
                rate = args.starting_rate
            else:
                rate = args.default_rate

            gen_loss_A_total = ((fm_loss*0.9+gen_loss*0.1)*(1.-rate))/2.0
            gen_loss_B_total = ((fm_loss*0.9+gen_loss*0.1)*(1.-rate))/2.0
            gen_loss_ReconA_total = (fm_loss_ReconB*0.9 + gen_loss_ReconB*0.1) * (1.-rate)/2.0
            gen_loss_ReconB_total = (fm_loss_ReconA*0.9 + gen_loss_ReconA*0.1) * (1.-rate)/2.0

            #############################################################################


            if args.model_arch == 'adrec_edges2shoes_ali_nospv_fm':
                gen_loss = gen_loss_A_total + gen_loss_B_total + gen_loss_ReconA_total + gen_loss_ReconB_total
                dis_loss = dis_loss + dis_loss_ReconA + dis_loss_ReconB
            elif args.model_arch == 'gan':
                gen_loss = gen_loss_B
                dis_loss = dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters % args.log_interval == 0:
                print( "---------------------")
                print( "GEN Loss:", as_np(gen_loss.mean()), as_np(gen_loss_ReconA.mean()), as_np(gen_loss_ReconB.mean()))
                print( "DIS Loss:", as_np(dis_loss.mean()), as_np(dis_loss_ReconA.mean()), as_np(dis_loss_ReconB.mean()))

            if iters % args.image_save_interval == 0:
                AB = generator_B( test_A )
                BA = generator_A( test_B )
                ABA = generator_A( AB )
                BAB = generator_B( BA )

                n_testset = min( test_A.size()[0], test_B.size()[0] )

                subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

                for im_idx in range( n_testset ):
                    A_val = test_A[im_idx].cpu().data.numpy().transpose(1,2,0) * 255.
                    B_val = test_B[im_idx].cpu().data.numpy().transpose(1,2,0) * 255.
                    BA_val = BA[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
                    ABA_val = ABA[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
                    AB_val = AB[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
                    BAB_val = BAB[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.

                    filename_prefix = os.path.join (subdir_path, str(im_idx))
                    scipy.misc.imsave( filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:,:,::-1])
                    scipy.misc.imsave( filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:,:,::-1])
                    scipy.misc.imsave( filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:,:,::-1])
                    scipy.misc.imsave( filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:,:,::-1])
                    scipy.misc.imsave( filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:,:,::-1])
                    scipy.misc.imsave( filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:,:,::-1])

            if iters % args.model_save_interval == 0:
                torch.save( generator_A, os.path.join(model_path, 'model_gen_A-' + str( iters / args.model_save_interval )))
                torch.save( generator_B, os.path.join(model_path, 'model_gen_B-' + str( iters / args.model_save_interval )))
                torch.save( discriminator_ali, os.path.join(model_path, 'model_dis_ali-' + str( iters / args.model_save_interval )))
                #torch.save( discriminator_spv_A, os.path.join(model_path, 'model_dis_spv_A-' + str( iters / args.model_save_interval )))
                #torch.save( discriminator_spv_B, os.path.join(model_path, 'model_dis_spv_B-' + str( iters / args.model_save_interval )))

            iters += 1

if __name__=="__main__":
    main()
