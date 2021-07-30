from simulator.radios.DtnBasicRadio import DtnBasicRadio


class DtnBasicRadioRL(DtnBasicRadio):

    def __init__(self, env, parent, shared=True):
        # Call parent constructor
        super().__init__(env, parent, shared)
        self.commanded_datarate = None

    def set_new_datarate(self, Tprop, new_datarate):
        self.env.process(self.do_set_new_datarate(Tprop, new_datarate))

    def do_set_new_datarate(self, Tprop, new_datarate):
        self.commanded_datarate = new_datarate

        yield self.env.timeout(Tprop)

        self.datarate = self.commanded_datarate
