import numpy as np
from src import read_pdb as pdb
from src.read_ligand import readLigandClusters


###################
# parameters
inputfile = "mean_analysis.dat"
pdbfile = "comp.pdb"
ligandClusterFile ="LIG_clusters.dat"
cutoff = 4  # distance cutoff for printing
output1 = "new.dat"  # printing data satisfying the cutoff
output = "new-distances.dat"
output2 = "select_res.dat"  # printing unique residue indices
output3 = "select_indices.dat"  # printing unique atom indices

###################


def getGroupIndex(index, groups):
    """
    Deprecated
    :param index:
    :param groups:
    :return:
    """
    for g in groups:
        if index in g:
            return groups.index(g)
    return None


def readData(filepath):
    """
    Columns are:
    [0]<ligand atom index>
    [1]<ligand resid>
    [2]<ligand atom type>
    [3]<protein atom index>
    [4]<protein resid>
    [5]<protein atom type>
    [6]<distance>
    [7]<protein resname>
    [8]<ligand resname>

    :param filepath:
    :return: list of frames indices [<frame>][<row>][<column>]
    """
    frames = []
    frame = -1
    numberOfLines = 0
    with open(filepath, 'r') as dat:
        for line in dat:
            numberOfLines += 1
            elements = line.split()
            if int(elements[0]) != frame:
                frames.append([])
                frame += 1
            frames[frame].append([int(elements[1]), int(elements[2]), elements[4], int(elements[5]), int(elements[6]), elements[8], np.float_(elements[9]), elements[7], elements[3]])

    print("{:d} lines read.".format(numberOfLines))
    return frames


class Pair:
    def __init__(self, line):
        self.sum = line[6]
        self.value = [line[6]]
        self.count = 1
        self.mean = None
        self.mask = None
        self.ligand_atom = {"index": line[0], "resid": line[1], "type": line[2], "resname": line[8]}
        self.ligandClusterAtoms = [self.ligand_atom]
        self.atom = {"index": line[3], "resid": line[4], "type": line[5], "resname": line[7]}
        self.proteinClusterAtoms = [self.atom]
        self.ID = -1
        return

    def __str__(self):
        return "{0:s}{1:d}{2:s}-{3:s}{4:d}{5:s}".format(self.atom["resname"],
                                                        self.atom["resid"],
                                                        self.atom["type"],
                                                        self.ligand_atom["resname"],
                                                        self.ligand_atom["resid"],
                                                        self.ligand_atom["type"])

    def calculateMean(self):
        self.sum = np.sum(self.value)
        self.mean = self.sum / self.count
        return

    def hasAtom(self, index):
        atoms = [self.atom["index"], self.ligand_atom["index"]]
        for a in self.proteinClusterAtoms:
            atoms.append(a["index"])
        for a in self.ligandClusterAtoms:
            atoms.append(a["index"])
        if index in atoms:
            return True
        else:
            return False

    def getProteinClusterAtoms(self, PDB):
        try:
            top = PDB.structure.top # old version, using PDB object
        except AttributeError:
            top = PDB # new version, get the topology from Cycle object
        protFragments = {
            "ARG": [{"NE", "NH1", "NH2", "CZ"}],
            "HID": [{"CG", "CE1", "ND1", "CD2", "NE2"}],
            "HIE": [{"CG", "CE1", "ND1", "CD2", "NE2"}],
            "HSD": [{"CG", "CE1", "ND1", "CD2", "NE2"}],
            "HSE": [{"CG", "CE1", "ND1", "CD2", "NE2"}],
            "HSP": [{"CG", "CE1", "ND1", "CD2", "NE2"}],
            "ASP": [{"CG", "OD1", "OD2"}],
            "GLU": [{"CD", "OE1", "OE2"}],
            "ASN": [{"CG", "OD1", "ND2"}],
            "GLN": [{"CD", "OE1", "NE2"}],
            "VAL": [{"CB", "CG1", "CG2"}],
            "ILE": [{"CB", "CG1", "CD", "CG2"}],
            "LEU": [{"CG", "CD1", "CD2"}],
            "PHE": [{"CG", "CE1", "CE2"}, {"CD1", "CD2", "CZ"}],
            "TYR": [{"CG", "CE1", "CE2"}, {"CD1", "CD2", "CZ"}],
            "TRP": [{"NE1", "CD2", "CZ2", "CZ3"}, {"CD1", "CE2", "CE3", "CH2"}]
        }
        if self.atom["resname"] in protFragments.keys():
            for frag in protFragments[self.atom["resname"]]:
                if self.atom["type"] in frag:
                    for i in top.select("residue "+str(self.atom["resid"])+" and resname "+str(self.atom["resname"])):
                        if top._atoms[i].name in frag:
                            if not self.hasAtom(i):
                                self.proteinClusterAtoms.append(
                                    {"index": i,
                                    "resid": self.atom["resid"],
                                    "type": top._atoms[i].name,
                                    "resname": self.atom["resname"]})
                            else:
                                pass
        return

    def getLigandClusterAtoms(self, PDB, ligFragments):
        try:
            top = PDB.structure.top # old version, using PDB object
        except AttributeError:
            top = PDB # new version, get the topology from Cycle object
        for frag in ligFragments:
            if self.ligand_atom["type"] in frag:
                selection = top.select(
                        "residue " + str(self.ligand_atom["resid"]) + " and resname "
                        + str(self.ligand_atom["resname"]))
                if len(selection) == 0:
                    print("WARNING: default selection does not find the ligand with resname {:s} and resid {:d}"
                          .format(self.ligand_atom["resname"], self.ligand_atom["resid"]))
                    print("Trying with resname...")
                    selection = top.select("resname "
                        + str(self.ligand_atom["resname"]))
                if len(selection) == 0:
                    print("WARNING: resname selection does not find the ligand with resname {:s}"
                          .format(self.ligand_atom["resname"]))
                    print("Trying with resid excluding water...")
                    selection = top.select("not water and residue " + str(self.ligand_atom["resid"]))
                if len(selection) == 0:
                    print("ERROR: atom {:s} was found in cluster {:s}, but residue could not be selected."
                          .format(self.ligand_atom, frag))
                for i in selection:
                    if top._atoms[i].name in frag:
                        if not self.hasAtom(i):
                            self.ligandClusterAtoms.append(
                                {"index": i,
                                 "resid": self.ligand_atom["resid"],
                                 "type": top._atoms[i].name,
                                 "resname": self.ligand_atom["resname"]})
        return

    def getPrintable(self):
        if self.proteinClusterAtoms == [] and self.ligandClusterAtoms == []:
            return "{0:d} {1:d} {2:s} {3:d} {6:s} {4:d} {5:s}\n".format(
                    self.ligand_atom["index"], self.ligand_atom["resid"], self.ligand_atom["type"],
                    self.atom["index"], self.atom["resid"], self.atom["type"], self.atom["resname"])
        elif self.proteinClusterAtoms != [] and self.ligandClusterAtoms == []:
            lines = ""
            for ap in self.proteinClusterAtoms:
                lines += "{0:d} {1:d} {2:s} {3:d} {6:s} {4:d} {5:s}\n".format(
                    self.ligand_atom["index"], self.ligand_atom["resid"], self.ligand_atom["type"],
                    ap["index"], ap["resid"], ap["type"], ap["resname"])
            return lines
        elif self.proteinClusterAtoms == [] and self.ligandClusterAtoms != []:
            lines = ""
            for al in self.ligandClusterAtoms:
                lines += "{0:d} {1:d} {2:s} {3:d} {6:s} {4:d} {5:s}\n".format(
                    al["index"], al["resid"], al["type"],
                    self.atom["index"], self.atom["resid"], self.atom["type"], self.atom["resname"])
            return lines
        else:
            lines = ""
            for al in self.ligandClusterAtoms:
                for ap in self.proteinClusterAtoms:
                    lines += "{0:d} {1:d} {2:s} {3:d} {6:s} {4:d} {5:s}\n".format(
                        al["index"], al["resid"], al["type"],
                        ap["index"], ap["resid"], ap["type"], ap["resname"])
            return lines


def createPairs(data):
    """
    :param data:
    :return:
    """
    groupID = []
    groups = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if (data[i][j][0], data[i][j][3]) in groupID:
                gi = groupID.index((data[i][j][0], data[i][j][3]))
                groups[gi].sum += data[i][j][6]
                groups[gi].value.append(data[i][j][6])
                groups[gi].count += 1
                groups[gi].mask.append(i)
            else:
                groupID.append((data[i][j][0], data[i][j][3]))
                groups.append(Pair(data[i][j]))
                groups[-1].mask = [i]
    return groups #, groupID


def printPairs(pairs):
    print("Printing groups:")
    lines2print = []
    means = []
    PDB = pdb.PDB()
    PDB.read(pdbfile)
    ligandClusters = readLigandClusters(ligandClusterFile)
    for i in range(len(pairs)):
        pairs[i].calculateMean()
        if pairs[i].mean < cutoff:
            pairs[i].getProteinClusterAtoms(PDB)
            pairs[i].getLigandClusterAtoms(PDB, ligandClusters)
            newline = pairs[i].getPrintable()
            if newline in lines2print:
                if pairs[i].mean < means[lines2print.index(newline)]:
                    means[lines2print.index(newline)] = pairs[i].mean
            else:
                lines2print.append(newline)
                means.append(pairs[i].mean)
    means, lines2print = zip(*sorted(zip(means, lines2print)))
    with open(output, 'w') as dat:
        for line in lines2print:
            dat.write("{:.6f}\n".format(means[lines2print.index(line)]))
            dat.write(line)
    with open(output1, 'w') as dat:
        for line in lines2print:
            dat.write(line)
    print("Done.")


def printSelection(data):
    print("Printing selections:")
    ind = set([])
    res = set([])
    for frame in data:
        ind = ind | set(np.array(frame)[:, 0])
        res = res | set(np.array(frame)[:, 1])
        ind = ind | set(np.array(frame)[:, 3])
        res = res | set(np.array(frame)[:, 4])
    with open(output2, 'w') as sel:
        for r in res:
            sel.write("{:d} ".format(int(r)))
    print("[1/2]")

    with open(output3, 'w') as sel:
        for i in ind:
            sel.write("{:d} ".format(int(i)))
    print("[2/2]")


def main():
    data = readData(inputfile)
    g = createPairs(data)
    printPairs(g)
    printSelection(data)


if __name__ == "__main__":
    main()
