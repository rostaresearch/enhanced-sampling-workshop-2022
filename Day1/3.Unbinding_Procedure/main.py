import argparse
import os
import sys

import src.cycle
import src.unbinding as unb
import src.read_pdb as rp
import src.output as out
from src.contact import UnboundException


class Arguments:
    def __init__(self, trajectory=None, lig="LIG", top="find", cutoff=3.5, maxdist=9, ns=10, cumulative=False,
                 writeDCD=False, stride=5, processonly=False, nosave=False, report=False, auto=False, namd=None,
                 maxiter=25, string=False):
        self.trajectory = trajectory
        self.lig = lig
        self.top = top
        self.cutoff = cutoff
        self.maxdist = maxdist
        self.ns = ns
        self.cumulative = cumulative
        self.writeDCD = writeDCD
        self.stride = stride
        self.processonly = processonly
        self.nosave = nosave
        self.report = report
        self.auto = auto
        self.namd = namd
        self.maxiter = maxiter
        self.string = string
        return


def run(args):
    Unb = unb.Unbinding()
    if args.report:
        if os.path.isfile(Unb.checkpoint):
            Unb = Unb.load()
            Unb.report()
        else:
            print("The is no checkpoint file to report of.")
    elif args.cumulative:
        if args.trajectory is None:
            print("With --cumulative option, please specify the last trajectory to be processed with option -t.")
            sys.exit(0)
        Unb.reprocess(args)
    else:
        if os.path.isfile(Unb.checkpoint):
            Unb = Unb.load()
            Unb.newCycle()
        else:
            Unb.readClusters()
            Unb.ligresname = args.lig
            Unb.traj_length = args.ns
            Unb.maxdist = args.maxdist / 10
            Unb.set_top(args.top)
            Unb.set_namd_template()
            Unb.cutoff = args.cutoff
        if args.trajectory is not None:
            Unb.cycle = int(args.trajectory) + 1
        c = src.cycle.Cycle(Unb)
        try:
            c.readDCD(Unb.top)
        except rp.DCDnotReadable:
            Unb.writeOutput("DCD file cannot be read for cycle {:d}".format(Unb.cycle - 1))
            sys.exit(0)
        if args.writeDCD:
            c.saveNewDCD(int(args.stride))
        c.getNeighbour(Unb.ligresname, Unb.cutoff, Unb.clusters)
        c.getClusters(Unb.clusters)
        Unb.history(c)
        c.createContact()
        if not args.nosave:
            out.trackDistances(Unb)
        if not args.processonly:
            # out.vmdRep(Unb)
            try:
                c.setupCycle()
                c.contact.writeNAMDcolvar(
                    os.path.join(c.wrkdir, "traj_{:d}".format(c.number), "sum_{:d}.col".format(c.number)),
                    traj_length=int(Unb.traj_length))
            except UnboundException:
                Unb.writeOutput("No contact remained in the colvar, ligand unbound!")
                sys.exit(0)
        if Unb.cycle == 1:
            Unb.writeOutput(out.header())
        Unb.writeOutput(out.cycle(c))
        if not args.nosave:
            Unb.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--trajectory")
    parser.add_argument('-l', "--lig", default="LIG")
    parser.add_argument("--top", default="find")
    parser.add_argument('-c', "--cutoff", default=3.5, type=float, help="Initial cutoff for identifying neighbours in A")
    parser.add_argument('-m', "--maxdist", default=9, type=float, help="distance for exclusion fo pairs in A")
    parser.add_argument('-ns', default=10, type=int)
    parser.add_argument("--cumulative", action='store_true', default=False,
                        help='Reprocess all the example up to the one specified by "-t". ')
    parser.add_argument('--writeDCD', action='store_true', default=False, help='write the strided DCD')
    parser.add_argument('-s', "--stride", default=5, type=int)
    parser.add_argument('-p', "--processonly", action='store_true', default=False,
                        help='Do not write VMD and NAMD input. Other outputs will be written.')
    parser.add_argument("--nosave", action='store_true', default=False,
                        help='Do not save the checkpoint. For debug only.')
    parser.add_argument("--report", action='store_true', default=False,
                        help='Report status and exit.')
    parser.add_argument("--auto", action='store_true', default=False,
                        help='Go to the background and restart after namd finished.')
    parser.add_argument("--namd", help="NAMD submission script, taking <name>.inp as an input and writing to"
                                       " <name>.out. Use with --auto.")
    parser.add_argument('--maxiter', default=25, type=int, help="Maximum number of iterations. Use with --auto.")
    parser.add_argument("--string", action='store_true', default=False)
    args = parser.parse_args()
    run(args)
