import argparse
import torch
from CNN.CNNModel.model.wide_res_net import WideResNet
from CNN.CNNModel.model.smooth_cross_entropy import smooth_crossentropy
from CNN.CNNModel.dataPreprocessing.fashion_mnist import Fashion_MNIST
from CNN.CNNModel.utility.log import Log
from CNN.CNNModel.utility.initialize import initialize
from CNN.CNNModel.utility.step_lr import StepLR
from CNN.CNNModel.utility.bypass_bn import enable_running_stats, disable_running_stats
import sys
sys.path.append("")
from CNN.CNNModel.sam import SAM
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=100, type=int,help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device check
    print(device)

    dataset = Fashion_MNIST(args.batch_size, args.threads)

    # ---------------- Setting the training proprieties ----------------
    log = Log(log_each=10)  # by default, it's = 10
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    # ------------- [TESTING] Print model's state_dict -------------
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # ------------- [TESTING] Print model's state_dict -------------

    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=args.rho,
        adaptive=args.adaptive,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    # ----- [UNCOMMENT TO ENABLE] Load the previous check point -----
    # PATH = r'./wide_resnet_state_dic.pt'
    # _CHECKPOINT = torch.load(f=PATH, map_location=device)
    # model.load_state_dict(state_dict=_CHECKPOINT)
    # x = torch.rand(size=(1, 3, 40, 40))
    # summary(model=model, input_data=x)
    # ----- [UNCOMMENT TO ENABLE] Load the previous check point -----

    # --------- Setting Update Threshold ---------
    GLOBAL_MAX_ACC = 95

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))


        def permute_add_channel(inputs: torch.Tensor):
            inputs = inputs.permute(1, 0, 2, 3)
            inputs = torch.cat((inputs, inputs, inputs), dim=0)
            inputs = inputs.permute(1, 0, 2, 3)
            return inputs

        _iter_batch = 0
        for batch in dataset.train:
            # ----- Count for test model per 300 batches -----
            _iter_batch += 1
            inputs, targets = (b.to(device) for b in batch)

            # ------ make the dataPreprocessing can be train by wide resnet ------
            inputs = permute_add_channel(inputs)

            # ------------- first forward-backward step ----------------
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # ------------- second forward-backward step ----------------
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

            # -------- Testing Block nested in training loop --------
            if (_iter_batch % 300 == 0):
                # --------- Enable evaluation mode for safe ---------
                model.eval()
                # --------- Set for update state_dict properties ---------
                _eval_correct = 0
                _eval_overall = 0
                # --------- Test per 300 batches ---------
                with torch.no_grad():
                    for eval_batch in dataset.test:
                        eval_inputs, eval_targets = (b.to(device) for b in eval_batch)

                        eval_inputs = permute_add_channel(eval_inputs)
                        eval_predictions = model(eval_inputs)
                        _correct = torch.argmax(eval_predictions, 1) == eval_targets
                        _eval_correct += _correct.sum()
                        _eval_overall += len(eval_targets)

                _accuracy = _eval_correct * 100.0 / _eval_overall
                if (_accuracy > GLOBAL_MAX_ACC):
                    GLOBAL_MAX_ACC = _accuracy
                    PATH = r'../../GUI/PYQTUtility/wide_resnet_state_dic.pt'
                    torch.save(model.state_dict(), PATH)
                    print("Save. Global_accuracy {}".format(GLOBAL_MAX_ACC))

                # -------- Shift back to training mode --------
                model.train()

        # ------ Same Setting as above ------
        eval_correct = 0
        eval_overall = 0
        # ------ Enable evaluation for safe ------
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                inputs = permute_add_channel(inputs)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                eval_correct += correct.sum()
                eval_overall += len(targets)

        # ------------- Save the model after evaluation ----------------
        eval_accuracy = eval_correct * 100.0 / eval_overall
        if eval_accuracy > GLOBAL_MAX_ACC:
            GLOBAL_MAX_ACC = eval_accuracy
            PATH = r'../../GUI/PYQTUtility/wide_resnet_state_dic.pt'
            torch.save(model.state_dict(), PATH)
            print("Save. Global_accuracy {}".format(GLOBAL_MAX_ACC))

        # ----- [REUSABLE MODULE: Load model for evaluation] -----
        # CHECK_POINT = torch.load(f=PATH, map_location=device)
        # RELOAD_MODE = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
        # RELOAD_MODE.load_state_dict(state_dict=CHECK_POINT, strict=True)
        # RELOAD_MODE.eval()
        # summary(model=RELOAD_MODE, input_data=(3, 40, 40))
        # ----- [REUSABLE MODULE: Load model for evaluation] -----

    log.flush()
