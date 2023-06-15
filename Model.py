from CoolProp.CoolProp import PropsSI
import constant
import abc

class SupplyChainNode():
    # instance counter
    next_id = 0

    def __init__(self, name, fix_cost):
        # unique name
        self.name = name + str(SupplyChainNode.next_id)
        SupplyChainNode.next_id += 1
        # name of the equipment ("Electrolyzer", "Compressor", "Dispenser", "Storage", "Buffer", "TubeTrailer")
        self.equipment = name
        # The echelon to which the node belongs (reference to the object of type SupplyChainEchelon)
        self.echelon = None
        # fix cost of the device
        self.fix_cost = fix_cost
        # list of next nodes
        self.next = []
        self.prev = []


class Receiver(SupplyChainNode):
    def __init__(self, name, fix_cost, pMin, pMax, tempMax, volume, racks, bottlesPerRack, P0, T0):
        super().__init__(name, fix_cost)
        # Min pressure in Pa
        self.pMin = pMin
        # Max pressure in Pa
        self.pMax = pMax
        # Max Temperature in K
        self.tempMax = tempMax
        # current Temperature in K
        self.temp = T0
        # Volume in L
        self.volume = volume
        # Number of racks (not implemented yet)
        self.racks = racks
        # Number of bottles per rack (not implemented yet)
        self.bottlesPerRack = bottlesPerRack 
        # initialize the pressure
        self.Pt = [P0]
        # initialize the compressibility factor
        self.Zt = [0]
        # initialize the mass of H2 in kg
        self.Mt = [0]


    # calculate the Compressibility factor at period t
    def calculate_Zt(self, t):
        # calculate density at given Temperature and pressure
        rho = PropsSI('D', 'T', self.temp, 'P', self.Pt[t], 'Hydrogen')
        # calculate the Compressibility factor
        self.Zt[t] = self.Pt[t] * constant.MOLAR_MASS_H2 / \
            (rho * constant.R * self.temp)

    # calculate the H2 mass at period t
    def calculate_Mt(self, t):
        if t == 0:
            self.Mt[t] = self.Pt[t]*self.volume * \
                constant.MOLAR_MASS_H2 / (constant.R * self.temp*self.Zt[t])
        else:
            if len(self.prev) > 0 and len(self.next) > 0:
                self.Mt[t] = self.Mt[t-1] + self.prev[0].Qt[t] - self.next[0].Qt[t]
            else:
                raise ValueError(f"In node {self.name} Next or Prev connection does not exists!")
    # calculate the final pressure at period t.
    def calculate_Pt(self, t):
        # t: period t
        self.Pt[t] = (self.Zt[t]*self.Mt[t]*constant.R *
                      self.temp)/(self.volume*constant.MOLAR_MASS_H2)

    # calculate the available capacity in H2 kg for a given pressure in period t
    def calculate_available_capacity(self, pressure, t):
        # Validate that the given pressure do not exceed the max pressure
        feasible_pressure = min(pressure, self.pMax)
        # calculate density at given Temperature and pressure
        rho = PropsSI('D', 'T', self.temp, 'P', feasible_pressure, 'Hydrogen')
        # calculate the Max Compressibility factor
        zFactor = feasible_pressure * constant.MOLAR_MASS_H2 / \
            (rho * constant.R * self.temp)
        # calculate the Max Capacity in kg (n*MOLAR_MASS_H2)
        maxCapacity = feasible_pressure * self.volume * \
            constant.MOLAR_MASS_H2 / (constant.R * self.temp*zFactor)
        # calculate the available Capacity in kg
        available_capacity = max(0, maxCapacity - self.Mt[t])
        return available_capacity


class TubeTrailer(Receiver):
    def __init__(self, name, fix_cost, pMin, pMax, tempMax, volume, racks, bottlesPerRack, P0, T0):
        super().__init__(name, fix_cost, pMin, pMax, tempMax, volume, racks, bottlesPerRack, P0, T0)
        # tracking the pipe or CompressorTT ID connected to the tube trailer in period t
        # ID equal 0 means that there is no pipe connected
        self.id_sender_t = []
        # Arrival time at the server (Pipe or CompressorTT)
        self.arrivalTime = 0

# update Tube trailer ID
    def update_id_tube_trailer_t(self):
        if len(self.prev):
            self.id_sender_t.append(self.prev[0].name)
        else:
            self.id_sender_t.append(None)

class Sender(SupplyChainNode, abc.ABC):
    def __init__(self, name, fix_cost, outputPressureMax):
        super().__init__(name, fix_cost)
        # Max output pressure in Pa
        self.outputPressureMax = outputPressureMax
        # output flow of hydrogen at period 0 in kg/h
        self.Qt = [0]

    @abc.abstractmethod
    def calculate_Qt(self, t):
        pass


class Electrolyzer(Sender):
    def __init__(self, name, fix_cost, outputPressureMax, Gt):
        super().__init__(name, fix_cost, outputPressureMax)
        # production in period t
        self.Gt = Gt

    def calculate_Qt(self, t):
        # initialize output flow at period t
        self.Qt[t] = 0
        # calculate the available capacity in the next receiver in kg given the max output pressure of the electrolizer in period t.
        if len(self.next) > 0:
            available_capacity = self.next[0].calculate_available_capacity(
                self.outputPressureMax, t-1)
            # validate the available capacity in the next receiver in kg
            if available_capacity > 0:
                # production rate constraint
                self.Qt[t] = min(available_capacity, self.Gt[t])
        else:
            raise ValueError(f"In node {self.name} Next connection does not exists!")


class Compressor(Sender):
    def __init__(self, name, fix_cost, outputPressureMax, inputPressureMin, inputPressureMax, slope, intercept):
        super().__init__(name, fix_cost, outputPressureMax)
        # min input pressure in Pa
        self.inputPressureMin = inputPressureMin
        # max input pressure in Pa
        self.inputPressureMax = inputPressureMax
        # compressor slope
        self.slope = slope
        # compressor intercept
        self.intercept = intercept

    def calculate_Qt(self, t):
        # initialize output flow at period t
        self.Qt[t] = 0
        # evaluate only if there is a previous receiver
        if(len(self.prev) > 0):
            # validate that the input pressure is feasible
            inputPressure = self.prev[0].Pt[t-1]
            if inputPressure >= self.inputPressureMin and inputPressure < self.inputPressureMax:
                # if the next sender is a Pipe and the Pipe is connected to a tube trailer
                try:
                    if self.next[0].next[0].equipment == "Pipe" and len(self.next[0].next[0].next) > 0:
                        # calculate the available capacity in the tube trailer in kg given the max output pressure of the compressor in period t.
                        available_capacity = self.next[0].next[0].next[0].calculate_available_capacity(
                            self.outputPressureMax, t-1)
                    else:
                        # calculate the available capacity in the next receiver in kg given the max output pressure of the compressor in period t.
                        available_capacity = self.next[0].calculate_available_capacity(
                            self.outputPressureMax, t-1)
                except IndexError:
                        raise ValueError(f"The flow is disconnected 2 or more connections ahead from node {self.name}!")
                
                # if there is available capacity to fulfill
                if available_capacity > 0:
                    # estimated the max flow rate
                    maxQ = self.slope * inputPressure + self.intercept
                    # never exceed the available capacity in the next receiver
                    self.Qt[t] = min(available_capacity, maxQ)


class CompressorTT(Compressor):
    def __init__(self, name, fix_cost, outputPressureMax, inputPressureMin, inputPressureMax, slope, intercept):
        super().__init__(name, fix_cost, outputPressureMax, inputPressureMin, inputPressureMax, slope, intercept)

    # evaluate at the end of the period if it is time to disconnect the Tube trailer from the pipe
    def unplug_tube_trailer(self, t):
        # evaluate only when a tube trailer is connected
        if len(self.prev) > 0:
            # check if the tube trailer's pressure
            input_pressure = self.prev[0].Pt[t]
            # if the tube trailer's pressure is less than the min input pressure in the compressor, proceed to disconnect
            if input_pressure < self.inputPressureMin:
                # reference to the tube trailer
                tubeTrailerRef = self.prev[0]
                # remove the connection between the CompressorTT and the tube trailer
                self.echelon.remove_edge(self.name, tubeTrailerRef.name)
                # remove the tube trailer from the echelon and backwards (node.echelon = None) 
                self.echelon.remove_node(tubeTrailerRef)
                # find a destination to the tube trailer, implement a function with multiple origins and destinations
                destination = next(iter(self.echelon.destinations))
                # calculate travel time, implement a function with multiple origins and destinations
                travelTime = constant.TRAVEL_TIME
                # update tube trailer's arrival time (NOTE: reconnection time could be implemented in a different way)
                tubeTrailerRef.arrivalTime = t + travelTime + constant.RECONNECTION_TIME
                # add tube trailer to the destination's Queue
                destination.queueTT.append(tubeTrailerRef)

    # evaluate at the end of the period t if there is an available Tube trailer to be connected into the CompressorTT
    def plug_tube_trailer(self, t):
        # evaluate only when a tube trailer is not connected
        if len(self.prev) == 0:
            # check if there is an available tube trailer waiting on the queue
            available_tube_trailer = None
            for i, trailer in enumerate(self.echelon.queueTT):
                if int(trailer['arrivalTime']) <= t:
                    # identify and remove the available tube trailer from the queue
                    available_tube_trailer = self.echelon.queueTT.pop(i)
                    break
            # if a tube trailer was found available in the queue proceed to connect it to the Pipe
            if available_tube_trailer:
                # add the tube trailer from the queue to the echelon list of nodes and update node.echelon
                self.echelon.add_node(available_tube_trailer)
                # add the connection between the Pipe and the tube trailer
                self.echelon.add_edge(available_tube_trailer.name, self.name)


class Dispenser(Sender):
    def __init__(self, name, fix_cost, outputPressureMax, inputPressureMin, Dt):
        super().__init__(name, fix_cost, outputPressureMax)
        # min input pressure in Pa
        self.inputPressureMin = inputPressureMin
        # demand in period t
        self.Dt = Dt

    def calculate_Qt(self, t):
        # if the initial input pressure is not greater than the min input pressure then output flow will be 0
        self.Qt[t] = 0
        if len(self.prev) > 0:
            inputPressure = self.prev[0].Pt[t-1]
            if inputPressure >= self.inputPressureMin:
                # supply the demand
                self.Qt[t] = self.Dt[t]
        else:
            raise ValueError(f"In node {self.name} Prev connection does not exists!")

class Pipe(Sender):
    def __init__(self, name, fix_cost, outputPressureMax, inputPressureMin):
        super().__init__(name, fix_cost, outputPressureMax)
        # min input pressure in Pa
        self.inputPressureMin = inputPressureMin
        # tracking of the tube trailers IDs connected to the pipe in period t
        # ID equal 0 means that there is no Tube trailer connected
        self.idTubeTrailer_t = []
    # update Tube trailer ID
    def update_id_tube_trailer_t(self):
        if len(self.next):
            self.idTubeTrailer_t.append(self.next[0].name)
        else:
            self.idTubeTrailer_t.append(None)

    # calculate the output flow in the Pipe
    def calculate_Qt(self, t):
        try:
            self.Qt[t] = self.prev[0].prev[0].calculate_Qt(t)
        except IndexError:
            raise ValueError(f"The flow is disconnected 2 or more connections behind the node {self.name}!")
        
    # evaluate at the end of the period if it is time to unplug the Tube trailer from the pipe
    def unplug_tube_trailer(self, t):
        # evaluate only when a tube trailer is connected
        if len(self.next) > 0:
            # check the tube trailer's available capacity
            available_capacity = self.next[0].calculate_available_capacity(
                self.next[0].Pt[t], t)
            # if the tube trailer has no available capacity proceed to disconnect
            if available_capacity == 0:
                # reference to the tube trailer
                tubeTrailerRef = self.next[0]
                # remove the connection between the Pipe and the tube trailer
                self.echelon.remove_edge(self.name, tubeTrailerRef.name)
                # remove the tube trailer from the echelon
                self.echelon.remove_node(tubeTrailerRef)
                # find a destination to the tube trailer, implement a function with multiple origins and destinations
                destination = next(iter(self.echelon.destinations))
                # calculate travel time, implement a function with multiple origins and destinations
                travelTime = constant.TRAVEL_TIME
                # update tube trailer's arrival time, NOTE: reconnection time can be implemented differently
                tubeTrailerRef.arrivalTime = t + travelTime + constant.RECONNECTION_TIME
                # add tube trailer to the destination's Queue
                destination.queueTT.append(tubeTrailerRef)

    # evaluate at the end of the period t if there is an available Tube trailer to be connected into the Pipe
    def plug_tube_trailer(self, t):
        # evaluate only when a tube trailer is not connected
        if len(self.next) == 0:
            # check if there is an available tube trailer waiting on the queue
            available_tube_trailer = None
            for i, trailer in enumerate(self.echelon.queueTT):
                if int(trailer['arrivalTime']) <= t:
                    # identify and remove the available tube trailer from the queue
                    available_tube_trailer = self.echelon.queueTT.pop(i)
                    break
            # if a tube trailer was found available in the queue proceed to connect it to the Pipe
            if available_tube_trailer:
                # add the tube trailer from the queue to the echelon list of nodes
                self.echelon.add_node(available_tube_trailer)
                # add the connection between the Pipe and the tube trailer
                self.echelon.add_edge(self.name, available_tube_trailer.name)


# An Echelon in the supply chain is represented by a Directed Acyclic Graph
class SupplyChainEchelon:
    next_id = 0

    def __init__(self, name):
        # Name of the echelon "FC" or "DC"
        self.name = name + str(SupplyChainEchelon.next_id)
        SupplyChainEchelon.next_id += 1
        # Initialize an empty dictionary to hold the graph nodes
        self.nodes = {}
        # list of destinations echelons
        self.destinations = {}
        # tube trailer queue
        self.queueTT = []

    # Method to add a echelon to the destinations

    def add_destination(self, echelon):
        if echelon.name not in self.destinations:
            self.destinations[echelon.name] = echelon
        else:
            raise ValueError(f"Node '{echelon.name}' already exists.")

    # Method to remove a node to the destinations
    def remove_destination(self, echelon_name):
        if echelon_name in self.destinations:
            del self.destinations[echelon_name]
        else:
            raise ValueError(f"Echelon '{echelon_name}' does not exist.")

    # Method to add a node to the graph
    def add_node(self, node):
        if node.name not in self.nodes:
            self.nodes[node.name] = node
            node.echelon = self
        else:
            raise ValueError(f"Node '{node.name}' already exists.")

    # Method to remove a node from the graph
    def remove_node(self, node):
        if node.name in self.nodes:
            del self.nodes[node.name]
            node.echelon = None
        else:
            raise ValueError(f"Node '{node.name}' does not exist.")

    # Method to add an edge to the graph
    def add_edge(self, node_name_from, node_name_to):
        if node_name_from in self.nodes and node_name_to in self.nodes:
            self.nodes[node_name_from].next.append(self.nodes[node_name_to])
            self.nodes[node_name_to].prev.append(self.nodes[node_name_from])
        else:
            raise ValueError("One or both nodes do not exist.")

    # Method to remove an edge from the graph
    def remove_edge(self, node_name_from, node_name_to):
        if node_name_from in self.nodes and node_name_to in self.nodes:
            self.nodes[node_name_from].next.remove(self.nodes[node_name_to])
            self.nodes[node_name_to].prev.remove(self.nodes[node_name_from])
        else:
            raise ValueError("One or both nodes do not exist.")

    # Method to get a node from the graph
    def get_node(self, node_name):
        if node_name in self.nodes:
            return self.nodes[node_name]
        else:
            raise ValueError(f"Node '{node_name}' does not exist.")

    # Method to get all nodes from the graph
    def get_all_nodes(self):
        return self.nodes

    # Method to validate the graph, ensuring it is a directed acyclic graph
    def validate(self):
        visited = set()
        for node in self.nodes.values():
            if not self.dfs(node, visited):
                return False
        return True

    # Depth-first search helper for the validate method
    def dfs(self, node, visited):
        if node.name in visited:
            return False
        visited.add(node.name)
        for child in node.next:
            if not self.dfs(child, visited):
                return False
        visited.remove(node.name)
        return True


class FillingCenter(SupplyChainEchelon):
    def __init__(self, name, Gt):
        super().__init__(name)
        self.Gt = Gt


class DischargingCenter(SupplyChainEchelon):
    def __init__(self, name):
        super().__init__(name)
        # min inventory level to trigger a new order
        self.reorder = 0
