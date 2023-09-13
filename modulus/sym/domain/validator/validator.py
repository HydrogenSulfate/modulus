import paddle


class Validator:
    """
    Validator base class
    """

    def forward_grad(self, invar):
        pred_outvar = self.model(invar)
        return pred_outvar

    def forward_nograd(self, invar):
        with paddle.no_grad():
            pred_outvar = self.model(invar)
        return pred_outvar

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        raise NotImplementedError(
            'Subclass of Validator needs to implement this')

    @staticmethod
    def _l2_relative_error(true_var, pred_var):
        new_var = {}
        for key in true_var.keys():
            new_var['l2_relative_error_' + str(key)] = paddle.sqrt(x=paddle
                .mean(x=paddle.square(x=true_var[key] - pred_var[key])) /
                paddle.var(x=true_var[key]))
        return new_var
