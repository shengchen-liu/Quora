from common import *
from config import *
import utils
from models.model import*
# import torchvision
# import torchvision.transforms.functional as f
# from torchvision import transforms as T

import argparse
#-----------------------------------------------
# Arg parser
# changes
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME", help="NAME OF OUTPUT FOLDER",
                    default="Baseline", type=str)
parser.add_argument("--MODE", help="TRAIN OR TEST",
                    default="train",type=str)
parser.add_argument("--INITIAL_CHECKPOINT", help="CHECK POINT",
                    type=str)
parser.add_argument("--RESUME", help="RESUME RUN",
                    type=bool)
parser.add_argument("--BATCH_SIZE", help="BATCH SIZE TIMES NUMBER OF GPUS",
                    default=10, type=int)
parser.add_argument("--GPUS", help="GPU", default='0',
                    type=str)
parser.add_argument("--LR", help="INITIAL LEARNING RATE",
                    default=1e-4,type=float)
parser.add_argument("--CHECKPOINT", help="CHECK POINT FOR TEST",
                    default=0, type=int)
parser.add_argument("--FOLD", help="KFOLD",
                    default=5, type=int)
parser.add_argument("--MODEL", help="BASE MODEL",
                    default="baseline", type=str)

args = parser.parse_args()

config = DefaultConfigs()
config.resume = args.RESUME
config.model_name = args.MODEL_NAME
config.initial_checkpoint = args.INITIAL_CHECKPOINT
config.batch_size = args.BATCH_SIZE
config.mode = args.MODE
config.gpus = args.GPUS
config.lr = args.LR
config.checkpoint = args.CHECKPOINT
config.fold = args.FOLD
config.model = args.MODEL

# 1. set random seed
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
try:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', os.environ['CUDA_VISIBLE_DEVICES'])
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =', 'None')
    NUM_CUDA_DEVICES = 1
warnings.filterwarnings('ignore')

if not os.path.exists('../results'):
    os.mkdir('../results')

if not os.path.exists(config.logs):
    os.mkdir(config.logs)

log = utils.Logger()
log.open('{0}{1}_log_train.txt'.format(config.logs, config.model_name),mode="a")
for arg in vars(args):
    log.write('{0}:{1}\n'.format(arg, getattr(args, arg)))
log.write("\n-------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                          |------ Train ------|------ Valid ------|----Best Results---|------------|\n')
log.write('mode    iter   epoch    lr|  loss    f1_macro |  loss    f1_macro |  loss    f1_macro | time       |\n')
log.write('----------------------------------------------------------------------------------------------------\n')

def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start, threshold=0.3):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        # compute output
        output = data_parallel(model,images)
        # output = model(images)
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))
        
        f1_batch = f1_score(target.cpu(),output.sigmoid().cpu() > threshold,average='macro')
        f1.update(f1_batch,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |  %0.3f   %0.3f    |   %0.3f    %0.4f   |  %s      %s       | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, f1.avg, 
                valid_loss[0], valid_loss[1], 
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [losses.avg,f1.avg]

# 2. evaluate fuunction
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start, threshold=0.3):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

            # optain output of a batch
            output = data_parallel(model, images_var)

            # Concatenate all every batch
            if i == 0:
                total_output = output
                total_target = target
            else:
                total_output = torch.cat([total_output, output], 0)
                total_target = torch.cat([total_target, target], 0)

        # compute loss for the entire evaluation dataset
        loss = criterion(total_output, total_target)
        losses.update(loss.item(), images_var.size(0))
        f1_batch = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > threshold, average='macro')
        f1.update(f1_batch, images_var.size(0))
        print('\r', end='', flush=True)
        message = '%s   %5.1f %6.1f        |  %0.3f   %0.3f    |   %0.3f    %0.4f   |  %s      %s       | %s' % ( \
            "val", epoch, epoch,
            train_loss[0], train_loss[1],
            losses.avg, f1.avg,
            str(best_results[0])[:8], str(best_results[1])[:8],
            time_to_str((timer() - start), 'min'))

        print(message, end='', flush=True)

        log.write("\n")

    return [losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,thresholds):
    sample_submission_df = pd.read_csv("./input/sample_submission.csv")
    #3.1 confirm the model converted to cuda
    filenames, labels, submissions= [],[],[]
    model.cuda()
    model.eval()
    submit_results = []

    def apply_transform(iter, loader):
        predictions = []
        for i, (input, filepaths) in enumerate(tqdm(loader)):
            # 3.2 change everything to cuda and get only basename
            filepaths = [os.path.basename(x) for x in filepaths]
            input = T.Compose([T.ToPILImage()])(input.squeeze())

            if iter < 4:#rotate
                input = f.rotate(input, 90*(iter+1))
            elif iter ==4: #hflip
                input = f.hflip(input)
            elif iter ==5: #vflip
                input = f.vflip(input)
            elif iter ==6: #resize 1.1
                input = f.affine(input, angle=0, translate=(0,0), shear=0,scale=1.1)
            elif iter ==7: #resize 1/1.1
                input = f.affine(input, angle=0, translate=(0,0), shear=0, scale=1/1.1)
            elif iter == 8:  # translate
                input = f.affine(input, angle=0, translate=(20,20), shear=0, scale=1)
            elif iter == 9:  # translate
                input = f.affine(input, angle=0, translate=(-20,-20), shear=0, scale=1)

            with torch.no_grad():
                input = f.to_tensor(input)
                input = input.unsqueeze(0)
                image_var = input.cuda(non_blocking=True)
                y_preds = model(image_var)
                # label = y_pred.sigmoid().cpu().data.numpy()
                y_preds = y_preds.cpu().data.numpy()

                for y_pred, filepath in zip(y_preds, filepaths):
                    predictions.append(y_pred)
                    if iter==0:
                        filenames.append(filepath)

        # predictions = np.array(predictions)
        return predictions

    results = []

    for i in range(6):
        print('TTA {}'.format(i))
        r = apply_transform(i,test_loader)
        results.append(r)

    results = np.array(results)
    results = results.mean(axis=0)


    for result in results:
        row = result > thresholds
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./results/submit/%s_submission.csv' % config.model_name, index=None)

# 4. main function
def main():

    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep + 'fold_'+str(config.fold)):
        os.makedirs(config.weights + config.model_name + os.sep + 'fold_'+ str(config.fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

    tqdm.pandas()

    start_time = time.time()
    train_X, test_X, train_y, word_index = utils.load_and_prec(config)
    # embedding_matrix_1 = load_glove(word_index)
    # embedding_matrix_2 = load_para(word_index)

    total_time = (time.time() - start_time) / 60
    print("Took {:.2f} minutes".format(total_time))

    # # embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis=0)
    # embedding_matrix = embedding_matrix_1
    #
    # # embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis=1)
    # print(np.shape(embedding_matrix))
    #
    # # del embedding_matrix_1, embedding_matrix_2
    # del embedding_matrix_1

    # -------------------------------------------------------
    # training
    # -------------------------------------------------------
    train_preds = np.zeros((len(train_X)))
    test_preds = np.zeros((len(test_X)))

    x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=config.batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train_X, train_y))

    for i, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()

        model = Baseline_Bidir_LSTM_GRU(config, word_index)
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        optimizer = torch.optim.Adam(model.parameters())

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        print(f'Fold {i + 1}')

        for epoch in range(config.epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_X))
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * config.batch_size:(i + 1) * config.batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, config.epochs, avg_loss, avg_val_loss, elapsed_time))

        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()

            test_preds_fold[i * config.batch_size:(i + 1) * config.batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)









    if config.mode == 'train':
        all_files = pd.read_csv(config.train_data)
        # oversample
        s = Oversampling("./input/train.csv")
        sample_names = all_files['Id']
        sample_names = [idx for idx in sample_names for _ in range(s.get(idx))]
        all_files = all_files.copy().set_index('Id')
        all_files = all_files.reindex(sample_names)
        all_files = all_files.rename_axis('Id').reset_index()

        for fold in range(config.fold):
            # 4.2 get model
            # model = get_net()
            if config.model == "baseline":
                model = get_net_resnet18()
            elif config.model == "resnet34":
                model = get_net_resnet34()

            model.cuda()

            optimizer = optim.Adam(model.parameters(), lr = config.lr)

            # ================================================================== #
            #                        Loss criterioin                             #
            # ================================================================== #
            # criterion
            # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)

            # Use the optim package to define an Optimizer that will update the weights of
            # the model for us. Here we will use Adam; the optim package contains many other
            # optimization algoriths. The first argument to the Adam constructor tells the
            # optimizer which Tensors it should update.
            assert config.loss in ['bcelog', 'f1_loss', 'focal_loss'], \
                print("Loss type {0} is unknown".format(config.loss))
            if config.loss == 'bcelog':
                criterion = nn.BCEWithLogitsLoss().cuda()
            elif config.loss == 'f1_loss':
                criterion = F1_loss().cuda()
            elif config.loss == 'focal_loss':
                criterion = FocalLoss().cuda()

            # best_loss = 999
            # best_f1 = 0
            best_results = [np.inf,0]
            val_metrics = [np.inf,0]

            ## k-fold--------------------------------

            # tflogger
            tflogger = TFLogger(os.path.join('results', 'TFlogs',
                                             config.model_name+"_fold{0}_{1}".format(config.fold, fold)))

            with open(os.path.join("./input/fold_{0}".format(config.fold),
                                   'train_fold{0}_{1}.txt'.format(config.fold, fold)), 'r') as text_file:
                train_names = text_file.read().split('\n')
                # # oversample
                # s = Oversampling("./input/train.csv")
                # train_names = [idx for idx in train_names for _ in range(s.get(idx))]
                train_data_list = all_files[all_files['Id'].isin(train_names)]
                # train_data_list = all_files.copy().set_index('Id')
                # train_data_list
                # train_data_list = train_data_list.reindex(train_names)
                # 57150 -> 29016
                # reset index
                # train_data_list = train_data_list.rename_axis('Id').reset_index()
            with open(os.path.join("./input/fold_{0}".format(config.fold),
                                   'test_fold{0}_{1}.txt'.format(config.fold, fold)), 'r') as text_file:
                val_names = text_file.read().split('\n')
                val_data_list = all_files[all_files['Id'].isin(val_names)]

            # #print(all_files)
            # # train_data_list,val_data_list = train_test_split(all_files,test_size = 0.13,random_state = 2050)
            #
            # # using a split that includes all classes in val
            # with open(os.path.join("./input/protein-trainval-split", 'tr_names.txt'), 'r') as text_file:
            #     train_names =  text_file.read().split(',')
            #     # oversample
            #     s = Oversampling("./input/train.csv")
            #     train_names = [idx for idx in train_names for _ in range(s.get(idx))]
            #     # train_data_list = all_files[all_files['Id'].isin(train_names)]
            #     train_data_list = all_files.copy().set_index('Id')
            #     # train_data_list
            #     train_data_list = train_data_list.reindex(train_names)
            #     #57150 -> 29016
            #     #reset index
            #     train_data_list = train_data_list.rename_axis('Id').reset_index()
            # with open(os.path.join("./input/protein-trainval-split", 'val_names.txt'), 'r') as text_file:
            #     val_names =  text_file.read().split(',')
            #     val_data_list = all_files[all_files['Id'].isin(val_names)]

            # load dataset
            train_gen = HumanDataset(train_data_list,config.train_data,mode="train")
            train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=4)

            val_gen = HumanDataset(val_data_list,config.train_data,augument=False,mode="train")
            val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=4)

            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=7, verbose=True)

            if config.resume:
                log.write('\tinitial_checkpoint = %s\n' % config.initial_checkpoint)
                checkpoint_path = os.path.join(config.weights, config.model_name, config.initial_checkpoint,'checkpoint.pth.tar')
                loaded_model = torch.load(checkpoint_path)
                model.load_state_dict(loaded_model["state_dict"])
                start_epoch = loaded_model["epoch"]
            else:
                start_epoch = 0

            scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
            start = timer()

            #train
            for epoch in range(start_epoch,config.epochs):
                scheduler.step(epoch)
                # train
                lr = get_learning_rate(optimizer)
                train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start,config.threshold)
                # val
                val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start, config.threshold)
                # check results
                is_best_loss = val_metrics[0] < best_results[0]
                best_results[0] = min(val_metrics[0],best_results[0])
                is_best_f1 = val_metrics[1] > best_results[1]
                best_results[1] = max(val_metrics[1],best_results[1])
                # save model
                save_checkpoint({
                    "epoch": epoch + 1,
                    "model_name": config.model_name,
                    "state_dict": model.state_dict(),
                    "best_loss": best_results[0],
                    "optimizer": optimizer.state_dict(),
                    "fold": config.fold,
                    "kfold": fold,
                    "best_f1": best_results[1],
                }, is_best_loss, is_best_f1, config.fold, fold)
                # print logs
                print('\r',end='',flush=True)

                log.write(
                    '%s  %5.1f %6.1f  %.2E|  %0.3f   %0.3f    |   %0.3f    %0.4f   |  %s      %s       | %s      |%s ' % ( \
                        "best", epoch, epoch, Decimal(lr),
                        train_metrics[0], train_metrics[1],
                        val_metrics[0], val_metrics[1],
                        str(best_results[0])[:8], str(best_results[1])[:8],
                        time_to_str((timer() - start), 'min'),
                        fold),
                )
                log.write("\n")
                time.sleep(0.01)

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {'Train_loss': train_metrics[0], 'Train_F1_macro': train_metrics[1],
                        'Valid_loss': val_metrics[0], 'Valid_F1_macro': val_metrics[1],
                        'Learnging_rate': lr}

                for tag, value in info.items():
                    tflogger.scalar_summary(tag, value, epoch)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tflogger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                    tflogger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
                # -------------------------------------
                # end tflogger

                # ================================================================== #
                #                        Early stopping                         #
                # ================================================================== #
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(val_metrics[0], model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break


    # -------------------------------------------------------
    # testing
    # -------------------------------------------------------
    elif config.mode=='test':
        test_files = pd.read_csv("./input/sample_submission.csv")
        test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
        test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

        # checkpoint_path = os.path.join(config.best_models,'{0}_fold_{1}_model_best_loss.pth.tar'.format(config.model_name, fold))
        checkpoint_path = os.path.join(config.weights, config.model_name, 'fold_{0}'.format(fold),
                                       'checkpoint_{}.pth.tar'.format(config.checkpoint))
        best_model = torch.load(checkpoint_path)
        #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
        model.load_state_dict(best_model["state_dict"])
        thresholds =[-0.13432257, -0.4642075,  -0.50726506, -0.49715518, -0.41125674,  0.11581507,
                     -1.0143597,  -0.18461785, -0.61600877, -0.47275479, -0.9142859,  -0.44323673,
                     -0.58404387, -0.22959213, -0.26110631, -0.43723898, -0.97624685, -0.44612319,
                     -0.4492785,  -0.56681327, -0.16156543, -0.12577745, -0.75476121, -0.91473052,
                      -0.53361931, -0.19337344, -0.0857145,  -0.45739976]

        # thresholds = [-0.27631527, -0.31156957, -0.61893745, -1.01863398, -0.3141709,  -0.14000374,
        #               -0.6285302,  -0.43241383, -1.60594984, -0.14425374, -0.03979607, -0.25717957,
        #               -0.84905692, -0.37668712,  1.3710663,  -0.11193908, -0.81109447,  0.72506607,
        #               -0.05454339, -0.47056617, -0.16024197, -0.44002794, -0.65929407, -1.00900269,
        #               -0.86197429, -0.12346229, -0.4946575,  -0.52420557]
        test(test_loader,model,thresholds)
        print('Test successful!')
if __name__ == "__main__":
    main()
