import numpy as np
import os


def header():
    h = """"""
    return h


def cycle(c):
    lines = """"""
    lines += "TRAJECTORY {:d}\n\n".format(c.number)
    lines += "Pairs used in this cycle:\n"
    for pair in c.pairs:
        if len(pair.ligandClusterAtoms) == 0 and len(pair.proteinClusterAtoms) == 0:
            lines += "{0:d} {1:d} {2:s} ".format(pair.ligand_atom["index"], pair.ligand_atom["resid"], pair.ligand_atom["type"])
            lines += "{0:d} {1:d} {2:s}\n".format(pair.atom["index"], pair.atom["resid"], pair.atom["type"])
        elif len(pair.ligandClusterAtoms) != 0 and len(pair.proteinClusterAtoms) == 0:
            index = np.inf
            for la in pair.ligandClusterAtoms:
                if la["index"] < index:
                    index = la["index"]
                    resid = la["resid"]
                    type = la["type"]
            lines += "{0:d} {1:d} {2:s} ".format(index, resid, type)
            lines += "{0:d} {1:d} {2:s}\n".format(pair.atom["index"], pair.atom["resid"], pair.atom["type"])
        elif len(pair.ligandClusterAtoms) == 0 and len(pair.proteinClusterAtoms) != 0:
            lines += "{0:d} {1:d} {2:s} ".format(pair.ligand_atom["index"], pair.ligand_atom["resid"],
                                                 pair.ligand_atom["type"])
            index = np.inf
            for pa in pair.proteinClusterAtoms:
                if pa["index"] < index:
                    index = pa["index"]
                    resid = pa["resid"]
                    type = pa["type"]
            lines += "{0:d} {1:d} {2:s}\n".format(index, resid, type)
        else:
            index = np.inf
            for la in pair.ligandClusterAtoms:
                if la["index"] < index:
                    index = la["index"]
                    resid = la["resid"]
                    type = la["type"]
            lines += "{0:d} {1:d} {2:s} ".format(index, resid, type)
            index = np.inf
            for pa in pair.proteinClusterAtoms:
                if pa["index"] < index:
                    index = pa["index"]
                    resid = pa["resid"]
                    type = pa["type"]
            lines += "{0:d} {1:d} {2:s}\n".format(index, resid, type)
    return lines


def trackDistances(Unb):
    legend = []
    with open(os.path.join(Unb.wrkdir, "distances_tracked.csv"), "w") as f:
        f.write("pairs:    ")
        for ID in range(1, Unb.N_pairs + 1):
            f.write(" {:02d} ".format(ID))
        f.write("\n")
        for c in range(len(Unb.pairs)):
            f.write("Traj_{:02d}   ".format(c + 1))
            for ID in range(1, Unb.N_pairs + 1):
                MATCH = False
                for p in Unb.pairs[c]:
                    if p.ID == ID:
                        f.write("  X ")
                        MATCH = True
                        if len(legend) < ID:
                            lindex = p.ligand_atom["index"]
                            lresid = p.ligand_atom["resid"]
                            ltype = p.ligand_atom["type"]
                            for la in p.ligandClusterAtoms:
                                if la["index"] < lindex:
                                    lindex = la["index"]
                                    lresid = la["resid"]
                                    ltype = la["type"]
                            pindex = p.atom["index"]
                            presid = p.atom["resid"]
                            ptype = p.atom["type"]
                            for pa in p.proteinClusterAtoms:
                                if pa["index"] < pindex:
                                    pindex = pa["index"]
                                    presid = pa["resid"]
                                    ptype = pa["type"]
                            legend.append("{0:02d}  {1:5d}  {2:5d}  {3:4s}  {4:5d}  {5:5d}  {6:4s}\n".format(ID, lindex, lresid+1, ltype, pindex, presid+1, ptype))
                        break
                if not MATCH:
                    f.write("  0 ")
            f.write("\n")
        f.write("\n")
        f.write("LEGEND\n")
        f.write("          Ligand             Protein\n")
        f.write("ID  index  resid  type  index  resid  type\n")
        for line in legend:
            f.write(line)
    return


def vmdRep(Unb):
    with open(os.path.join(Unb.wrkdir, "representation_{:d}.tcl".format(Unb.cycle - 1)), "w") as f:
        f.write("mol new {:s} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n".format(Unb.top))
        for c in range(Unb.cycle):
            f.write("mol addfile traj_{0:d}/traj_{0}.dcd type dcd first 0 last -1 step 5 filebonds 1 autobonds 1 waitfor all\n".format(c))
            f.write("pbc set [readxst traj_{0:d}/traj_{0}.xst -step2frame {0} -first 0 -last -1 -stride 5]\n".format(c))

        f.write("""package require pbctools
pbc wrap -centersel "protein" -center com -compound residue -all
set reference [atomselect top "protein" frame 0]
set compare [atomselect top "protein" ]
set allcompare [atomselect top "all" ]
set num_steps [molinfo top get numframes]
    for {set frame 0} {$frame < $num_steps} {incr frame} {
        $compare frame $frame
        $allcompare frame $frame
        set trans_mat [measure fit $compare $reference]
        $allcompare move $trans_mat
	}

mol representation NewCartoon 0.300000 10.000000 4.100000 0
mol delrep 0 0
axes location Off
mol selection {all }
mol addrep top
mol representation Licorice 0.300000 10.000000 10.000000
mol selection {""" + "resname {0:s}".format(Unb.ligresname) + """}
mol addrep top
mol representation Lines
mol selection {not water and same residue as within 4 of """ + "resname {0:s}".format(Unb.ligresname) + """}
mol addrep top\n""")
        atoms = []
        pairs = []
        indices = ""
        for p in Unb.pairs[-1]:
            lindex = p.ligand_atom["index"]
            for la in p.ligandClusterAtoms:
                if la["index"] < lindex:
                    lindex = la["index"]
            pindex = p.atom["index"]
            for pa in p.proteinClusterAtoms:
                if pa["index"] < pindex:
                    pindex = pa["index"]
            atoms.append(pindex)
            pairs.append((lindex, pindex))
            for a in atoms:
                indices += "{:d} ".format(a)
        f.write("""mol representation CPK
mol selection {""" + "index {:s}".format(indices) + """ }
mol addrep top
""")
        for p in pairs:
            f.write("label add Bonds 0/{0:d} 0/{1:d}\n".format(p[0], p[1]))

    return
