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
parser.add_argument("--INITIAL_CHECKPOINT", help="CHECK POINT",
                    type=str)
parser.add_argument("--RESUME", help="RESUME RUN",
                    type=bool)
parser.add_argument("--BATCH_SIZE", help="BATCH SIZE TIMES NUMBER OF GPUS",
                    default=10, type=int)
parser.add_argument("--GPUS", help="GPU", default='0',
                    type=str)
parser.add_argument("--LR", help="INITIAL LEARNING RATE",
                    default=1e-3,type=float)
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
config.gpus = args.GPUS
config.lr = args.LR
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
log.write('                          |------ Train ------|------ Valid ------|------------|\n')
log.write('mode    iter   epoch    lr|       loss        |       loss        | time       |\n')
log.write('------------------------------------------------------------------------------- \n')

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

def train(train_loader,model,loss_fn, optimizer,epoch,valid_loss,start):
    losses = utils.AverageMeter()
    model.train()

    for i, (x_batch, y_batch) in enumerate(train_loader):
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(),x_batch.size(0))

        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f        |  %0.3f  |   %0.3f   | %s' % ( \
            "train", i / len(train_loader) + epoch, epoch,
            losses.avg,
            valid_loss,
            utils.time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)
    log.write("\n")
    return losses.avg


# 2. evaluate fuunction
def evaluate(val_loader,model,loss_fn,epoch,train_loss,start_time):
    losses = utils.AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(val_loader):
            y_pred = model(x_batch)
            # Concatenate all every batch
            if i == 0:
                total_output = y_pred
                total_target = y_batch
            else:
                total_output = torch.cat([total_output, y_pred], 0)
                total_target = torch.cat([total_target, y_batch], 0)

        # compute loss for the entire evaluation dataset
        print("total_output:", total_output.shape)
        print("total_target:", total_target.shape)
        
        val_loss = loss_fn(total_output, total_target)
        losses.update(val_loss.item(),total_target.shape[0])
        
        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f        |  %0.3f  |   %0.3f   | %s' % ( \
            "val", epoch, epoch,
            train_loss,
            losses.avg,
            utils.time_to_str((timer() - start_time), 'min'))
        print(message, end='', flush=True)

        log.write("\n")

    return losses.avg, sigmoid(total_output).cpu().data.numpy()[:, 0]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model):
    model.cuda()
    model.eval()
    predictions = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch)
            y_preds = sigmoid(y_pred).cpu().data.numpy()[:, 0]
            for y_pred in y_preds:
                predictions.append(y_pred)
    return np.array(predictions)


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
test_dataset = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train_X, train_y))

sigmoid = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

# k-fold
for fold, (train_idx, valid_idx) in enumerate(splits):
    print(f'Fold {fold + 1}')

    # tflogger
    tflogger = utils.TFLogger(os.path.join('results', 'TFlogs',
                                     config.model_name + "_fold{0}_{1}".format(config.fold, fold)))
    # initialize the early_stopping object
    early_stopping = utils.EarlyStopping(patience=7, verbose=True)

    x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    model = Baseline_Bidir_LSTM_GRU(config, word_index)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_dataset = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid_dataset = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    valid_loss = np.inf
    start_time = timer()
    for epoch in range(config.epochs):
        # train
        lr = utils.get_learning_rate(optimizer)
        train_loss = train(train_loader=train_loader,model=model,loss_fn=loss_fn, optimizer=optimizer,
                           epoch=epoch,valid_loss=valid_loss,start=start_time)

        # validate
        valid_loss, valid_output = evaluate(val_loader=valid_loader, model=model, loss_fn=loss_fn, epoch=epoch,
                                            train_loss=train_loss, start_time=start_time)
        test_preds_fold = np.zeros(len(test_X))

        # save model
        utils.save_checkpoint({
            "epoch": epoch,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "fold": config.fold,
            "kfold": config.fold,
        },config.fold, fold, config)
        # print logs
        print('\r', end='', flush=True)

        log.write("\n")
        time.sleep(0.01)

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'Train_loss': train_loss,
                'Valid_loss': valid_loss,
                'Learnging_rate': lr}

        for tag, value in info.items():
            tflogger.scalar_summary(tag, value, epoch)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            tflogger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            if not value.grad is None:
                tflogger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
        # -------------------------------------
        # end tflogger

        # ================================================================== #
        #                        Early stopping                         #
        # ================================================================== #
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # end looping all epochs
    train_preds[valid_idx] = valid_output
    # test
    test_preds_fold = test(test_loader=test_loader, model=model)
    test_preds += test_preds_fold / len(splits)

# end k-fold
search_result = threshold_search(train_y, train_preds)
print(search_result)

sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = test_preds > search_result['threshold']
sub.to_csv("submission_{0}.csv".format(config.model_name), index=False)

print('Test successful!')
if __name__ == "__main__":
    main()
