from crossflow.kernels import SubprocessKernel
from crossflow.filehandling import FileHandler
import mdtraj as mdt
import numpy as np
from pathlib import Path
import json
import itertools

class Walker(object):
    '''
    A WE simulation walker
    '''
    iterator = itertools.count()


    def __init__(self, state, weight, state_id=None):
        self.state = state
        self.weight = weight
        self.pc = None
        self.bin = None
        self.history = []
        self.state_id = state_id
        if self.state_id is None:
            self.state_id = next(Walker.iterator)
        self.data = {}
        
    def copy(self):
        new = Walker(self.state, self.weight, state_id=self.state_id)
        new.pc = self.pc
        new.bin = self.bin
        new.history = self.history.copy()
        new.data = self.data.copy()
        return new

    def update(self, state):
        self.history.append(self.state_id)
        self.state_id = next(Walker.iterator)
        self.state = state
        self.pc = None
        self.bin = None

    def __repr__(self):
        return ('<WElib.Walker weight {}, progress coordinate {}, bin assignment {}>'.format(self.weight, self.pc, self.bin))

class CrossflowGACheckpointer(object):
    '''
    A simple checkpointing class for Crossflow filehandles, Gromacs/Amber sims

    Saves coordinates and metadata for a list of walkers in a specified
    directory
    '''
    def __init__(self, dirname, mode='r'):
        self.dirname = Path(dirname)
        self.mode = mode
        if not 'w' in self.mode:
            if not self.dirname.exists():
                raise OSError('Error - checkpoint directory not found')
        self.dirname.mkdir(parents=True, exist_ok=True)

    def save(self, walkers):
        if not 'w' in self.mode:
            raise OSError('Error: checkpoint directory is read-only')
        metadata = {}
        for i,w in enumerate(walkers):
            name = 'walker_{:05d}'.format(i)
            metadata[name] = {}
            metadata[name]['weight'] = w.weight
            metadata[name]['state_id'] = w.state_id
            metadata[name]['history'] = w.history
            coordinates = w.state
            if hasattr(coordinates, 'result'):
                coordinates = w.state.result()
            crdfile = self.dirname / name
            if hasattr(coordinates, 'save'):
                if hasattr(coordinates, 'uid'):
                    ext = Path(coordinates.uid).suffix
                    crdfile = self.dirname / (name + ext)
                coordinates.save(crdfile)
            else:
                coordinates = Path(coordinates)
                crdfile = self.dirname / (name + coordinates.suffix)
                crdfile.write_bytes(coordinates.read_bytes())

        metadatafile = self.dirname / '_metadata_'
        with metadatafile.open('w') as f:
            json.dump(metadata, f)

    def load(self):
        metadatafile = self.dirname / '_metadata_'
        if not metadatafile.exists():
            raise OSError('Error: no metadata file found')
        with metadatafile.open() as f:
            metadata = json.load(f)
        crdfiles = list(self.dirname.glob('walker_*'))
        crdfiles.sort()
        walkers = []
        for i, name in enumerate(metadata):
            crds = crdfiles[i]
            wt = metadata[name]['weight']
            state_id = metadata[name]['state_id']
            w = Walker(crds, wt, state_id=state_id)
            w.history = metadata[name]['history']
            walkers.append(w)
        return walkers
        
class Recorder(object):
    def __init__(self):
        self.states = {}

    def record(self, walkers):
        for w in walkers:
            self.states[w.state_id] = w.state

    def replay(self, walker):
        statelist = []
        for i in walker.history:
            statelist.append(self.states[i])
        statelist.append(walker.state)
        return statelist

class FunctionStepper(object):
    # Move the walkers according to a supplied function
    def __init__(self, function, *args):
        self.function = function
        self.args = args
        self.recorder = Recorder()

    def run(self, walkers):
        self.recorder.record(walkers)
        for w in walkers:
            state = self.function(w.state, *self.args)
            w.update(state)
        self.recorder.record(walkers)
        return walkers

class CrossflowFunctionStepper(object):
    # Move the walkers according to a supplied function
    def __init__(self, client, function, *args):
        self.client = client
        self.function = function
        self.args = args
        self.recorder = Recorder()
    def run(self, walkers):
        self.recorder.record(walkers)
        states = [w.state for w in walkers]
        newstates = self.client.map(self.function, states, *self.args)
        for w, s in zip(walkers, newstates):
            w.update(s.result())
        self.recorder.record(walkers)
        return walkers

class CrossflowPMEMDCudaStepper(object):
    '''
    A WE simulation stepper
    
    Moves each walker a step forward (by running a bit of MD. Uses pmemd
    via crossflow

    Initialised with a crossflow client, and Amber mdin and prmtop files.
    '''
    def __init__(self, client, mdin, prmtop):
        self.client = client
        self.mdin = mdin
        self.prmtop = prmtop
        self.pmemd = SubprocessKernel('pmemd.cuda -i mdin -c in.ncrst -p x.prmtop -r out.ncrst -o mdlog -AllowSmallBox')
        self.pmemd.set_inputs(['mdin', 'in.ncrst', 'x.prmtop'])
        self.pmemd.set_outputs(['out.ncrst', 'mdlog'])
        self.pmemd.set_constant('mdin', mdin)
        self.pmemd.set_constant('x.prmtop', prmtop)
        self.recorder = Recorder()
        
    def run(self, walkers):
        self.recorder.record(walkers)
        inpcrds = [w.state for w in walkers]
        restarts, logs = self.client.map(self.pmemd, inpcrds)
        state_ids = [w.state_id for w in walkers]
        next_state_id = max(state_ids) + 1
        new_walkers = []
        for i, r in enumerate(restarts):
            if r.status == 'error':
                new_walkers.append(None)
            else:
                state = r.result()
                w = walkers[i]
                w.update(state)
                new_walkers.append(w)
        for i, w in enumerate(new_walkers):
            w.data['log'] = logs[i].result()
        self.recorder.record(new_walkers)
        return new_walkers

class FunctionProgressCoordinator(object):
    def __init__(self, pcfunc, *args):
        self.pcfunc = pcfunc
        self.args = args

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        for w in walkers:
            w.pc = self.pcfunc(w.state, *self.args)
        return walkers

class GAPCVProgressCoordinator(object):
    '''
    A WE simulation progress coordinate calculator

    Adds progress coordinate data to a list of walkers

    Initialised with a trajectory of the points that define the path,
    the indices of the atoms in the collective variable, and the PCV
    lambda parameter
    '''
    def __init__(self, pcv_traj, atom_indices, l):
        self.topology = pcv_traj.topology
        self.pcv_traj = pcv_traj.atom_slice(atom_indices)
        self.atom_indices = atom_indices
        self.l = l

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.topology)
        p = len(self.pcv_traj)
        s = []
        z = []
        x_l = walker_traj.atom_slice(self.atom_indices)
        for i, c in enumerate(x_l):
            rmsd = mdt.rmsd(self.pcv_traj, c)
            msd = rmsd * rmsd
            v = np.exp(msd * -self.l)
            vi = v * range(p)
            s = vi.sum() / ((p-1) * v.sum())
            z = -1/(self.l * np.log(v.sum()))
            walkers[i].pc = s
            walkers[i].data['z'] = z
        return walkers

class GASimpleDistanceProgressCoordinator(object):
    '''
    A WE simulation progress coordinate calculator
    
    Adds progress coordinate data to a list of walkers
    
    Initialised with an MDTraj Topology for the system and indices
    of the atoms to monitor the distance between.
    '''
    def __init__(self, topfile, atom_pair):
        self.topology = mdt.load_topology(topfile)
        self.atom_pair = atom_pair
        
    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.topology)
        pcs = mdt.compute_distances(walker_traj, [self.atom_pair])[:,0]
        for i, pc in enumerate(pcs):
            walkers[i].pc = pc
        return walkers
    

class GARMSDProgressCoordinator(object):
    '''
    A WE simulation progress coordinate generator

    Returns the RMSD of a set of atoms from a reference structure
    '''
    def __init__(self, ref, fit_sel):
        self.ref = ref
        self.fit_atoms = self.ref.topology.select(fit_sel)

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.ref.topology)
        pcs = mdt.rmsd(walker_traj, self.ref, atom_indices=self.fit_atoms)
        for i in range(len(walker_traj)):
            walkers[i].pc = pcs[i]
        return walkers

class GARMSD2ProgressCoordinator(object):
    '''
    A WE simulation progress coordinate generator

    Returns the fractional distance of a structure (1) between two
    reference points (2 & 3): f = r12/(r12+r13)
    '''
    def __init__(self, refstart, refend, fit_sel):
        self.refstart = refstart
        self.refend = refend
        self.fit_atoms = self.refstart.topology.select(fit_sel)

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.refstart.topology)
        rms12 = mdt.rmsd(walker_traj, self.refstart, atom_indices=self.fit_atoms)
        rms13 = mdt.rmsd(walker_traj, self.refend, atom_indices=self.fit_atoms)
        for i in range(len(walker_traj)):
            pc = rms12[i]/(rms12[i] + rms13[i])
            walkers[i].pc = pc
        return walkers

class StaticBinner(object):
    '''
    A WE simulation bin classifier
    
    Adds bin ID information to a list of walkers

    IDs are integers if there is just one dimension of binning,
    or tuples if more.
    
    Initialised with a list of the bin edges.
    '''
    def __init__(self, edges):
        self.edges = np.atleast_2d(edges)
        self.ndim = self.edges[0].ndim
        self.bin_weights = {}
        
    
    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        pcs = np.atleast_2d([w.pc for w in walkers]).T
        if None in pcs:
            raise TypeError('Error: missing progress coordinates...')
        bin_ids = []
        for dim in range(self.ndim):
            bin_ids.append(np.digitize(pcs[:, dim], self.edges[dim]))
        if self.ndim > 1:
            bin_ids = [z for z in zip(*bin_ids)]
        else:
            bin_ids = bin_ids[0]
        for i, bin_id in enumerate(bin_ids):
            walkers[i].bin = bin_id
            if not bin_id in self.bin_weights:
                self.bin_weights[bin_id] = 0.0
            self.bin_weights[bin_id] += walkers[i].weight
        sorted_dict = {}
        for key in sorted(self.bin_weights):
            sorted_dict[key] = self.bin_weights[key]
        self.bin_weights = sorted_dict
        return walkers

    def reset(self):
        for k in self.bin_weights:
            self.bin_weights[k] = 0.0
    
class MinimalAdaptiveBinner(object):
    '''
    Implements the minimal adaptive binning strategy
    '''
    def __init__(self, n_bins, retrograde=False):
        self.n_bins = n_bins
        self.retrograde = retrograde

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        n_walkers = len(walkers)
        for i in range(n_walkers):
            if walkers[i].pc is None:
                raise TypeError('Error - missing progress coordinate')
        walkers.sort(key=lambda w: w.pc)
        bin_width = (walkers[-1].pc - walkers[0].pc) / self.n_bins
        pc_min = walkers[0].pc
        if self.retrograde:
            walkers.reverse()
        w = np.array([w.weight for w in walkers])
        zmax = -np.log(w.sum())
        bottleneck = 0
        for i in range(n_walkers):
            walkers[i].bin = int((walkers[i].pc - pc_min) / bin_width)
            if i < n_walkers - 1:
                z = np.log(w[i]) - np.log(w[i+1:].sum())
                if z > zmax:
                    zmax = z
                    bottleneck = i
        walkers[0].bin = 'lag'
        walkers[-1].bin = 'lead'
        walkers[bottleneck].bin = 'bottleneck'
        return walkers

class Recycler(object):
    '''
    A WE simulation recycler
    
    Moves walkers that have reached the target pc back to the start
    Reports the recycled flux as well.
    
    Initialised with a copy of the initial state, and the
    value of the target pc. If retrograde == False, then recycling happens if
    target_pc is exceeded, else reycling happens if
    the pc falls below target_pc.
    '''
    def __init__(self, initial_state, target_pc, retrograde=False):
        self.initial_state = initial_state
        self.target_pc = target_pc
        self.retrograde = retrograde
        self.recycled_walkers = []
        self.flux_history = []
        self.flux = None
        
    def run(self, walkers):
        self.recycled_walkers = []
        self.flux = 0.0
        if not isinstance(walkers, list):
            walkers = [walkers]
        for i in range(len(walkers)):
            if walkers[i].pc is None:
                raise TypeError('Error - missing progress coordinate')
            recycle = False
            if not self.retrograde:
                recycle = walkers[i].pc > self.target_pc
            else:
                recycle = walkers[i].pc < self.target_pc
            if recycle:
                self.recycled_walkers.append(walkers[i])
                weight = walkers[i].weight
                walkers[i] = Walker(self.initial_state, weight)
                self.flux += weight 
        self.flux_history.append(self.flux)
        return walkers

class Bin(object):
    '''
    A WE simulation bin object - only used internally
    '''
    def __init__(self, index):
        self.index = index
        self.walkers = []
        
    def add(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        self.walkers += walkers
        
    def split_merge(self, target_size):
        if len(self.walkers) == target_size:
            ids = range(target_size)
        else:
            probs = np.array([w.weight for w in self.walkers])
            old_weight = probs.sum()
            ids = np.random.choice(range(len(self.walkers)), 
                    target_size, p=probs/old_weight)
        new_walkers = []
        for i in ids:
            new_walker = self.walkers[i].copy()
            new_walkers.append(new_walker)
        if len(self.walkers) != target_size:
            new_weight = np.array([w.weight for w in new_walkers]).sum()
            fac = old_weight / new_weight
            for i in range(len(new_walkers)):
                new_walkers[i].weight *= fac
        self.walkers = list(new_walkers)
        
class SplitMerger(object):
    '''
    A WE simulation splitter and merger
    
    Splits or merges the walkers in each bin to get to the required number of replicas
    
    Initialised with the desired number of replicas in each bin.
    '''
    def __init__(self, target_size):
        self.target_size = target_size
    
    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
            
        bins = {}
        for w in walkers:
            if not w.bin in bins:
                bins[w.bin] = Bin(w.bin)
            bins[w.bin].add(w)
        
        for bin in bins:
            bins[bin].split_merge(self.target_size)
        
        new_walkers = []
        for bin in bins:
            new_walkers += bins[bin].walkers
            
        return new_walkers

class OMMStepper(object):
    """
    An OpenMM MD stepper

    """
    def __init__(self, simulation, nsteps):
        self.simulation = simulation
        self.nsteps = nsteps

    def run(self, walkers):
        new_walkers = []
        for w in walkers:
            self.simulation.context.setPositions(w.state.getPositions())
            self.simulation.context.setPeriodicBoxVectors(*w.state.getPeriodicBoxVectors())
            self.simulation.step(self.nsteps)
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
            w.update(state)
            new_walkers.append(w)
            
        return new_walkers

class OMMSimpleDistanceProgressCoordinator(object):
    '''
    A WE simulation progress coordinate calculator for OpenMM sims
    
    Adds progress coordinate data to a list of walkers
    
    Initialised with indices
    of the atoms to monitor the distance between.
    '''

    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        
    def run(self, walkers):
        import openmm.unit as unit
        if not isinstance(walkers, list):
            walkers = [walkers]
        for i, w in enumerate(walkers):
            crds = w.state.getPositions(asNumpy=True)
            dx = crds[self.a1] - crds[self.a2]
            r = dx * dx
            pc = r.sum().sqrt() / unit.nanometer
            walkers[i].pc = pc
        return walkers
