import paddle


class Inferencer:
    """
    Inferencer base class
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
            'Subclass of Inferencer needs to implement this')
