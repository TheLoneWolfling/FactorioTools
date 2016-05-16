#!/usr/bin/env pypy3

#Copyright (C) 2016 James Harris <jharris@unb.ca>
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.

# This is a "little tool" for checking belt balancers in Factorio.

# In Factorio you have "splitters". They take two belts
# as input, and have two output belts.

# The question is: given some configuration of splitters, inputs, and outputs,
# is it "throughput-unlimited", and is it a true balancer?
# In other words, no matter what combination of inputs and outputs connected
# will it carry as much throughput as the inputs and outputs allow?

# This does the braindead-stupid method of just looping over all possible
# combinations of inputs and outputs active and solving the max-flow problem
# for each of them. Works well up to about n=8 or so.

# TODO: find a smarter way of doing this.

import itertools
import collections
import sys
import argparse

verbose = 3

VERSION = (1, 0, 0)
VERSION_STRING = ".".join(map(str, VERSION))

# Collections.deque is overkill
# Takes the "queue from one stack popped into the other when empty" approach
class Queue:
    def __init__(self, iterable=None):
        self.head = []
        self.tail = []
        if iterable is not None:
            for x in iterable:
                self.push(x)
    def push(self, val):
        self.head.append(val)
    def pop(self):
        if len(self.tail) == 0:
            self.tail.extend(reversed(self.head))
            self.head = []
            #self.head.clear() # not in pypy?
        return self.tail.pop()
    def isEmpty(self):
        return len(self.head) == 0 == len(self.tail)

class Edge:
    def __init__(self, nA, nB, cap=1):
        self.nA = nA
        self.nB = nB
        self.cap = cap
    def __str__(self):
        return '"{}" -> "{}" @ {}'.format(self.nA, self.nB, self.cap)
    def othNode(self, node):
        if node == self.nA:
            return self.nB
        elif node == self.nB:
            return self.nA
        
    def getCapTo(self, node, flows):
        if node == self.nA:
            return flows[self]
        elif node == self.nB:
            return self.cap - flows[self]
    def addResidFlow(self, node, flows, flow):
        if node == self.nA:
            flows[self] -= flow
        elif node == self.nB:
            flows[self] += flow

class Graph:
    def __init__(self):
        self.conns = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: None
            ))
        self.rconns = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: None
            ))
        self.nodeSet = set()
        self.__nodes = None
        self.numEdges = 0

        self.inputs = set()
        self.outputs = set()

    @property
    def nodes(self):
        if self.__nodes is None:
            self.__nodes = tuple(sorted(self.nodeSet))
        return self.__nodes

    @property
    def numNodes(self):
        return len(self.nodeSet)

    def __str__(self):
        toRet = ["Graph"]
        left = list(self.nodes)
        soFar = Queue(sorted(self.inputs))
        found = set(self.inputs)
        while len(left) > 0:
            while not soFar.isEmpty():
                n = soFar.pop()
                toRet.append('  "{}"'.format(n))
                for edge, othNode in self.getEdgesFrom(n):
                    toRet.append("    {}".format(edge))
                    if othNode not in found:
                        soFar.push(othNode)
                        found.add(othNode)
            n = left.pop()
            if n not in found:
                soFar.push(n)
                found.add(n)
        return "\n".join(toRet)

    def getEdge(self, a, b):
        return self.conns[a][b]

    def getEdgesFrom(self, node):
        return ((edge, oth) for (oth, edge) in self.conns[node].items())
    
    def getEdgesTo(self, node):
        return ((edge, oth) for (oth, edge) in self.rconns[node].items())

    def getEdgesFromOrTo(self, node):
        toRet = []
        for edge, oth in self.getEdgesFrom(node):
            toRet.append((edge, oth, True))
        for edge, oth in self.getEdgesTo(node):
            toRet.append((edge, oth, False))
        return toRet

    def addEdge(self, a, b, cap=1):
        if self.getEdge(a, b) is not None:
            raise ValueError("Edge already present: {}".format(self.getEdge(a, b)))
        self.conns[a][b] = self.rconns[b][a] = Edge(a, b, cap)
        self.addNode(a)
        self.addNode(b)
        self.numEdges += 1

        self.inputs.discard(b)
        self.outputs.discard(a)

    def removeEdge(self, edge):
        del self.conns[edge.nA][edge.nB]
        del self.rconns[edge.nB][edge.nA]

        if len(self.conns[edge.nA]) == 0:
            self.outputs.add(edge.nA)
            
        if len(self.rconns[edge.nB]) == 0:
            self.inputs.add(edge.nB)

        self.numEdges -= 1
        

    def addNode(self, n):
        if n not in self.nodeSet:
            self.nodeSet.add(n)
            self.inputs.add(n)
            self.outputs.add(n)
            self.__nodes = None
            return True
        return False

    def removeNode(self, n):
        for edge, _, _ in self.getEdgesFromOrTo(n):
            self.removeEdge(edge)
        self.nodeSet.remove(n)
        self.inputs.discard(n)
        self.outputs.discard(n)
        self.__nodes = None
        
    def maxFlow(self, nA, nB):
        flows = collections.defaultdict(int)
        
        flow = self.excess(nB, flows)

        while True:
            isFeasible, parents = self.getAugmentingPath(nA, nB, flows)
            if not isFeasible:
                break
            thisFlow = float("+inf")
            n = nB
            while n != nA:
                thisFlow = min(thisFlow, parents[n].getCapTo(n, flows))
                n = parents[n].othNode(n)

            n = nB
            while n != nA:
                parents[n].addResidFlow(n, flows, thisFlow)
                n = parents[n].othNode(n)

            flow += thisFlow

        return flow, flows

    def getAugmentingPath(self, nA, nB, flows):
        parents = dict()

        queue = Queue()
        queue.push(nA)
        while not queue.isEmpty():
            node = queue.pop()
            for edge, othNode, _ in self.getEdgesFromOrTo(node):
                if othNode in parents:
                    continue
                if edge.getCapTo(othNode, flows) > 0:
                    parents[othNode] = edge
                    if othNode != nB:
                        queue.push(othNode)
                    else:
                        return True, parents
        return False, parents

    def excess(self, node, flows):
        excess = 0
        for edge, othNode, isFrom in self.getEdgesFromOrTo(node):
            if isFrom:
                excess -= flows[edge]
            else:
                excess += flows[edge]
        return excess

class LineFormatError(ValueError):
    def __init__(self, error, errorCode=255):
        super("Invalid line format ({}) - needs to be of the form\n'inputID( outputID(=<number>)?)+'".format(error))
        this.errorCode = errorCode

def readLine(graph, line):
    lineNoNewline = line.rstrip()
    if lineNoNewline == "":
        return False
    if " " not in lineNoNewline:
        raise LineFormatError("No space found", 255)
    split = lineNoNewline.split(" ")
    inn = split[0]
    for out in split[1:]:
        if "=" in out:
            out, _, capStr = out.partition("=")
            try:
                cap = float(capStr)
            except ValueError:
                raise LineFormatError('"{}" is not a valid number'.format(capStr), 253)
        else:
            cap = 1
        if len(out) == 0:
            raise LineFormatError('no identifier before "="', 254)
        graph.addEdge(inn, out)
    return True

def readData(file=sys.stdin):
    try:
        graph = Graph()
        for line in file:
            if not readLine(graph, line):
                    break
        if verbose > 0:
            print("Read {} connections between {} nodes with {} inputs and {} outputs".format(
                        graph.numEdges, graph.numNodes, len(graph.inputs), len(graph.outputs)))
        if verbose > 1:
            print(graph)
        return graph
    finally:
        try:
            file.close()
        except IOError as e:
            if verbosity > 2:
                print("Ignoring error while closing file: {}".format(e))
            pass
        

# From Python docs.
def powerset(iterable, start=0):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1))


def testFlow(graph, activeInputs, activeOutputs):
    synthStartNode = object()

    synthEndNode = object()

    for inp in activeInputs:
        graph.addEdge(synthStartNode, inp)

    for out in activeOutputs:
        graph.addEdge(out, synthEndNode)

    flow, data = graph.maxFlow(synthStartNode, synthEndNode)

    graph.removeNode(synthStartNode)
    graph.removeNode(synthEndNode)

    return flow, data

def testAllFlows(graph):
    numTests = 0
    numSucc = 0
    inputSet = tuple(sorted([sorted(x) for x in powerset(graph.inputs, start=1)]))
    outputSet = tuple(sorted([sorted(x) for x in powerset(graph.outputs, start=1)]))
    numTests = len(inputSet) * len(outputSet)
    i = 0
    for inpSub in reversed(inputSet):
        for outSub in reversed(outputSet):
            shouldFlow = min(len(inpSub), len(outSub))
            numStatusUpdates = 100
            shouldPrint = (i+1) * numStatusUpdates // numTests > i * numStatusUpdates // numTests
            i += 1
            if verbose and shouldPrint:
                print("{}% done; testing {} -> {}, expecting {}...".format(
                    i * 100 * 100 // numTests / 100, "".join(inpSub), "".join(outSub), shouldFlow), end="")
                sys.stdout.flush()
            actFlow, data = testFlow(graph, inpSub, outSub)
            if verbose and shouldPrint:
                print(" got {}".format(actFlow))
            if actFlow != shouldFlow:
                print("{} -> {} got {} but expected {}!".format(
                    "".join(inpSub), "".join(outSub), actFlow, shouldFlow))
                for k, v in data.items():
                    print({k:v})
                return False
            numSucc += 1
    print("{} tests made, {} tests succeeded: LGTM (TM)!".format(numTests, numSucc))
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=
"""A simple utility for checking if a Factorio belt balancer is throughput-limited.

Expects graph inputs of the form of a bunch of lines of the following form

inputNodeID( outputNodeID(=<throughput>)?)+

For instance:

A B=1 D=2
B C D
C D
D C

With standard input, either use EOF or enter a blank line to finish.
"""
                                     )
    parser.add_argument("-V", "--version", action='version', version="%(prog)s {}".format(VERSION_STRING))
    parser.add_argument('FILE', nargs="?", default=sys.stdin, type=argparse.FileType('r'),
                        help=
                        "File to read from (standard input if omitted)")
    args = parser.parse_args()
    
    graph = readData(args.FILE)
    testAllFlows(graph)
            
    
