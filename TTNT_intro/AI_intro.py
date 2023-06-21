# Graph:

# V: {A1:(21.018703, 105.849342), A2:(21.018592377602225, 105.84998641458411),
#     A3:(21.01849221352082, 105.85044890789604), A4:(21.018275, 105.851601),
#     B1:(21.018034154786033, 105.85088308089018), C1:(21.01801361996291, 105.85039501216207),
#     C2:(21.017868631764006, 105.85084747208495), D1:(21.017685820355744, 105.85084409551837),
#     D2:(21.017621, 105.851466), E1:(21.016939, 105.849248),
#     E2:(21.016904141454468, 105.84982099584178), E3:(21.016856862357695, 105.8503308573967),
#     E4:(21.016809583245923, 105.85083058925184), E5:(21.016771, 105.851401), E6: (21.017267, 105.851407)
#     F1:(21.015125, 105.849202), F2:(21.015120133108727, 105.8498615146401),
#     F3:(21.01510437322296, 105.85036462306184), F4:(21.015101221245587, 105.85094201594853),
#     F5:(21.015114, 105.851573), G1:(21.013362, 105.849124),
#     T1: (21.014699, 105.849552), T2: (21.014697998012895, 105.84986625397741)
#     T3: (21.013885, 105.849153), T4: (21.013851, 105.849573)
#     G2:(21.01342120781558, 105.84989190373867), G3:(21.01346218397757, 105.85043215439289),
#     G4:(21.01351892018328, 105.8510601957784), G5:(21.013565, 105.851724)}
#     H1: (21.011821814072594, 105.8498486339589), H2: (21.011794, 105.850373)
#     H3: (21.011801, 105.850431), H4: (21.011775, 105.850949)

# Ma trận kề:
# A1 -> A2, E1
# A2 -> A1, A3
# A3 -> A2, A4, C1
# A4 -> A3
# C1 -> C2,E3
# C2 -> C1,B1,D1
# B1 -> C2
# D1 -> C2, D2
# D2 -> D1, A4
# E1 -> E2, F1
# E2 -> E1, A2, E3
# E3 -> E2, F3, E4
# E4 -> E3, E5, F4
# E5 -> E4, E6
# F1 -> F2, T3
# F2 -> F1, E2, F3
# F3 -> F2, F4, G3
# F4 -> F3, F5, E4, G4
# F5 -> F4, E5
# T1 -> T2
# T2 -> T1, F2
# T3 -> G1, T4
# T4 -> T3
# G1 -> G2
# G2 -> G3, F2
# G3 -> G4, H3
# G4 -> G5, F4, H4
# G5 -> F5
# H1 -> G2
# H2 -> H1
# H3 -> H2
# H4 -> H3, G4

from queue import PriorityQueue
from tkinter import messagebox
from tkintermapview import TkinterMapView
import networkx as nx
import heapq
import math
import customtkinter
import time

customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    APP_NAME = "TkinterMapView with CustomTkinter"
    WIDTH = 800
    HEIGHT = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(App.APP_NAME)
        self.geometry(str(App.WIDTH) + "x" + str(App.HEIGHT))
        self.minsize(App.WIDTH, App.HEIGHT)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.createcommand("tk::mac::Quit", self.on_closing)

        self.marker_list = []
        self.graph = nx.DiGraph()

        # ============ create two CTkFrames ============

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(
            master=self, width=150, corner_radius=0, fg_color=None
        )
        self.frame_left.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        self.frame_right = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_right.grid(row=0, column=1, rowspan=1, pady=0, padx=0, sticky="nsew")

        # ============ frame_left ============

        self.frame_left.grid_rowconfigure(2, weight=1)

        self.button_2 = customtkinter.CTkButton(
            master=self.frame_left,
            text="Clear Markers",
            command=self.clear_marker_event,
        )
        self.button_2.grid(pady=(20, 0), padx=(20, 20), row=0, column=0)

        self.test_mode_label = customtkinter.CTkLabel(
            self.frame_left, text="Choose Algorithm:", anchor="w"
        )

        self.test_mode_label.grid(row=1, column=0, padx=(20, 20), pady=(20, 0))

        self.test = customtkinter.CTkOptionMenu(
            self.frame_left,
            values=[
                "Find Shortest Path By A*",
                "Find Shortest Path By BFS",
                "Find Shortest Path By Dijkstra",
                "Find Shortest Path By Bellman-Ford",
                "Find Shortest Path By DFS",
            ],
            command=self.change_test,
        )

        self.test.grid(row=2, column=0, padx=(20, 20), pady=(10, 0))

        self.map_label = customtkinter.CTkLabel(
            self.frame_left, text="Tile Server:", anchor="w"
        )
        self.map_label.grid(row=3, column=0, padx=(20, 20), pady=(20, 0))
        self.map_option_menu = customtkinter.CTkOptionMenu(
            self.frame_left,
            values=["OpenStreetMap", "Google normal", "Google satellite"],
            command=self.change_map,
        )
        self.map_option_menu.grid(row=4, column=0, padx=(20, 20), pady=(10, 0))

        self.appearance_mode_label = customtkinter.CTkLabel(
            self.frame_left, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_label.grid(row=5, column=0, padx=(20, 20), pady=(20, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(
            self.frame_left,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode_optionemenu.grid(
            row=6, column=0, padx=(20, 20), pady=(10, 20)
        )
        self.map_option_menu.set("OpenStreetMap")
        self.appearance_mode_optionemenu.set("Dark")

        # ============ frame_right ============

        self.frame_right.grid_rowconfigure(1, weight=1)
        self.frame_right.grid_rowconfigure(0, weight=0)
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(1, weight=0)
        self.frame_right.grid_columnconfigure(2, weight=1)

        self.map_widget = TkinterMapView(self.frame_right, corner_radius=0)
        self.map_widget.grid(
            row=1,
            rowspan=1,
            column=0,
            columnspan=3,
            sticky="nswe",
            padx=(0, 0),
            pady=(0, 0),
        )
        self.map_widget.add_right_click_menu_command(
            label="Add Marker", command=self.add_marker_event, pass_coords=True
        )

        self.map_widget.set_polygon(
            [
                (21.018703, 105.849342),
                (21.018275, 105.851601),
                (21.01806668479017, 105.85143879689274),
                (21.01808505504749, 105.85124097073884),
                (21.017763952974764, 105.85115051965606),
                (21.01352360118644, 105.8513361789923),
                (21.013365313953724, 105.84935444961675),
                (21.013856173434995, 105.84934167724315),
                (21.01387941679973, 105.84970552864505),
                (21.014900330514823, 105.84969139282796),
                (21.01489619957613, 105.8493929618231),
                (21.015508159979817, 105.84943722411796),
                (21.01551942706826, 105.84961022659868),
                (21.016959206902932, 105.84965705167441),
                (21.016933543225814, 105.84984212409564),
                (21.018311839595363, 105.84989627307031),
                (21.018359764771134, 105.84935303323992),
            ],
            fill_color=None,
            outline_color="red",
            border_width=5,
            name="bui_thi_xuan_polygon",
        )

        self.entry = customtkinter.CTkEntry(
            master=self.frame_right, placeholder_text="type address"
        )
        self.entry.grid(row=0, column=0, sticky="we", padx=(12, 0), pady=12)
        self.entry.bind("<Return>", self.search_event)

        self.button_8 = customtkinter.CTkButton(
            master=self.frame_right, text="Search", width=90, command=self.search_event
        )
        self.button_8.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=12)

        # Set default values
        self.map_widget.set_address(
            "Nguyen Du, Ha Noi, Vietnam", text="Phuong Bui Thi Xuan, Ha Noi"
        )

        # ===================Thiết lập đồ thị===========================

        self.nodes = [
            (21.018703, 105.849342),
            (21.018592377602225, 105.84998641458411),
            (21.01849221352082, 105.85044890789604),
            (21.018275, 105.851601),
            (21.018034154786033, 105.85088308089018),
            (21.01801361996291, 105.85039501216207),
            (21.017868631764006, 105.85084747208495),
            (21.017685820355744, 105.85084409551837),
            (21.017621, 105.851466),
            (21.016939, 105.849248),
            (21.016904141454468, 105.84982099584178),
            (21.016856862357695, 105.8503308573967),
            (21.016809583245923, 105.85083058925184),
            (21.016771, 105.851401),
            (21.015125, 105.849202),
            (21.015120133108727, 105.8498615146401),
            (21.01510437322296, 105.85036462306184),
            (21.015101221245587, 105.85094201594853),
            (21.015114, 105.851573),
            (21.013362, 105.849124),
            (21.01342120781558, 105.84989190373867),
            (21.01346218397757, 105.85043215439289),
            (21.01351892018328, 105.8510601957784),
            (21.013565, 105.851724),
            (21.011821814072594, 105.8498486339589),
            (21.011794, 105.850373),
            (21.011801, 105.850431),
            (21.011775, 105.850949),
            (21.017267, 105.851407),
            (21.014699, 105.849552),
            (21.014697998012895, 105.84986625397741),
            (21.013885, 105.849153),
            (21.013851, 105.849573)
        ]

        self.G = {
            (21.018703, 105.849342): [  # A1
                (21.018592377602225, 105.84998641458411),  # A2
                (21.016939, 105.849248)  # E1
            ],
            (21.018592377602225, 105.84998641458411): [  # A2
                (21.018703, 105.849342),  # A1
                (21.01849221352082, 105.85044890789604),  # A3
            ],
            (21.01849221352082, 105.85044890789604): [  # A3
                (21.018592377602225, 105.84998641458411),  # A2
                (21.018275, 105.851601),  # A4
                (21.01801361996291, 105.85039501216207),  # C1
            ],
            (21.018275, 105.851601): [  # A4
                (21.01849221352082, 105.85044890789604)  # A3
            ],
            (21.01801361996291, 105.85039501216207): [  # C1
                (21.017868631764006, 105.85084747208495),  # C2
                (21.016856862357695, 105.8503308573967),  # E3
            ],
            (21.017868631764006, 105.85084747208495): [  # C2
                (21.01801361996291, 105.85039501216207),  # C1
                (21.018034154786033, 105.85088308089018),  # B1
                (21.017685820355744, 105.85084409551837),  # D1
            ],
            (21.018034154786033, 105.85088308089018): [  # B1
                (21.017868631764006, 105.85084747208495)  # C2
            ],
            (21.017685820355744, 105.85084409551837): [  # D1
                (21.017868631764006, 105.85084747208495),  # C2
                (21.017621, 105.851466),  # D2
            ],
            (21.017621, 105.851466): [  # D2
                (21.017685820355744, 105.85084409551837),  # D1
                (21.018275, 105.851601)  # A4
            ],
            (21.016939, 105.849248): [  # E1
                (21.016904141454468, 105.84982099584178),  # E2
                (21.015125, 105.849202)  # F1
            ],
            (21.016904141454468, 105.84982099584178): [  # E2
                (21.016939, 105.849248),  # E1
                (21.018592377602225, 105.84998641458411),  # A2
                (21.016856862357695, 105.8503308573967),  # E3
            ],
            (21.016856862357695, 105.8503308573967): [  # E3
                (21.016904141454468, 105.84982099584178),  # E2
                (21.01510437322296, 105.85036462306184),  # F3
                (21.016809583245923, 105.85083058925184),  # E4
            ],
            (21.016809583245923, 105.85083058925184): [  # E4
                (21.016856862357695, 105.8503308573967),  # E3
                (21.016771, 105.851401),  # E5
                (21.015101221245587, 105.85094201594853),  # F4
            ],
            (21.016771, 105.851401): [  # E5
                (21.016809583245923, 105.85083058925184),  # E4
                (21.017267, 105.851407)  # E6
            ],
            (21.017267, 105.851407): [  # E6
                (21.017621, 105.851466)  # D2
            ],
            (21.015125, 105.849202): [  # F1
                (21.015120133108727, 105.8498615146401),  # F2
                (21.013885, 105.849153)  # T3
            ],
            (21.015120133108727, 105.8498615146401): [  # F2
                (21.015125, 105.849202),  # F1
                (21.016904141454468, 105.84982099584178),  # E2
                (21.01510437322296, 105.85036462306184),  # F3
            ],
            (21.01510437322296, 105.85036462306184): [  # F3
                (21.015120133108727, 105.8498615146401),  # F2
                (21.015101221245587, 105.85094201594853),  # F4
                (21.01346218397757, 105.85043215439289),  # G3
            ],
            (21.015101221245587, 105.85094201594853): [  # F4
                (21.01510437322296, 105.85036462306184),  # F3
                (21.016809583245923, 105.85083058925184),  # E4
                (21.01351892018328, 105.8510601957784),  # G4
            ],
            (21.015114, 105.851573): [  # F5
                (21.015101221245587, 105.85094201594853),  # F4
                (21.016771, 105.851401)  #E5
            ],
            (21.014699, 105.849552): [  # T1
                (21.014697998012895, 105.84986625397741)  # T2
            ],
            (21.014697998012895, 105.84986625397741): [  # T2
                (21.014699, 105.849552),  # T1
                (21.015120133108727, 105.8498615146401)  # F2
            ],
            (21.013885, 105.849153): [  # T3
                (21.013851, 105.849573),  # T4
                (21.013362, 105.849124)  # G1
            ],
            (21.013851, 105.849573): [  # T4
                (21.013885, 105.849153)  # T3
            ],

            (21.013362, 105.849124): [  # G1
                (21.01342120781558, 105.84989190373867)  # G2
            ],
            (21.01342120781558, 105.84989190373867): [  # G2
                (21.01346218397757, 105.85043215439289),  # G3
                (21.014697998012895, 105.84986625397741)  # T2
            ],
            (21.01346218397757, 105.85043215439289): [  # G3
                (21.01351892018328, 105.8510601957784),  # G4
                (21.011801, 105.850431),  # H3
            ],
            (21.01351892018328, 105.8510601957784): [  # G4
                (21.013565, 105.851724),  # G5
                (21.015101221245587, 105.85094201594853),  # F4
                (21.011775, 105.850949),  # H4
            ],
            (21.013565, 105.851724): [(21.015114, 105.851573)],  # G5  # F5
            (21.011821814072594, 105.8498486339589): [  # H1
                (21.01342120781558, 105.84989190373867)  # G2
            ],
            (21.011794, 105.850373): [  # H2
                (21.011821814072594, 105.8498486339589)  # H1
            ],
            (21.011801, 105.850431): [(21.011794, 105.850373)],  # H3  # H2
            (21.011775, 105.850949): [  # H4
                (21.011801, 105.850431),  # H3
                (21.01351892018328, 105.8510601957784),  # G4
            ],
        }

    def make_graph(self):
        self.graph.clear()
        self.graph.add_nodes_from(self.nodes)
        for node in list(self.G.keys()):
            for neighbor in self.G[node]:
                self.graph.add_edge(
                    node, neighbor, weight=self.calculate_distance(node, neighbor)
                )

    # ============================Các hàm tính toán===============================

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_heuristic(self, point1, point2):
        return self.calculate_distance(point1, point2)

    def calculate_projection(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return line_start

        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))

        projection_x = x1 + t * dx
        projection_y = y1 + t * dy

        return (projection_x, projection_y)

    def calculate_path_cost(self, graph, path):
        cost = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            cost += graph[u][v]["weight"]
        return cost

    # =========================Các thuật toán tìm kiếm==============================

    def a_star(self, graph, start, goal):
        open_set = []
        came_from = {}
        g_scores = {}

        f_scores = {node: float("inf") for node in graph.nodes}
        f_scores[start] = self.calculate_heuristic(start, goal)

        heapq.heappush(open_set, (f_scores[start], start))
        g_scores[start] = 0

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, self.calculate_path_cost(graph, path)

            for neighbor in graph.neighbors(current):
                g_score = (
                    g_scores[current] + graph.get_edge_data(current, neighbor)["weight"]
                )

                if g_score < g_scores.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_scores[neighbor] = g_score
                    f_score = g_score + self.calculate_heuristic(neighbor, goal)
                    f_scores[neighbor] = f_score
                    heapq.heappush(open_set, (f_score, neighbor))

        return None, None

    def bfs(self, graph, start, goal):
        came_from = {}
        queue = []

        queue.append(start)
        came_from[start] = None

        while queue:
            current = queue.pop(0)

            if current == goal:
                path = []
                current = goal

                while current is not None:
                    path.append(current)
                    current = came_from[current]

                path.reverse()

                return path, self.calculate_path_cost(graph, path)

            for neighbor in graph.neighbors(current):
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current

        return None, None

    def dijkstra(self, graph, start, goal):
        came_from = {}
        open_set = PriorityQueue()
        open_set.put((0, start))

        g_score = {node: float("inf") for node in graph.nodes}
        g_score[start] = 0

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                path = [current]

                while current != start:
                    current = came_from[current]
                    path.append(current)

                path.reverse()
                return path, self.calculate_path_cost(graph, path)

            for neighbor in graph.neighbors(current):
                tentative_g_score = (
                    g_score[current] + graph[current][neighbor]["weight"]
                )
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    open_set.put((g_score[neighbor], neighbor))

        return None, None

    def bellman_ford(self, graph, start_node, target_node):
        dist = {node: float("inf") for node in graph.nodes}
        prev = {node: None for node in graph.nodes}
        dist[start_node] = 0

        for _ in range(len(graph.nodes) - 1):
            for u, v in graph.edges:
                weight = graph[u][v]["weight"]
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u

        for u, v in graph.edges:
            weight = graph[u][v]["weight"]
            if dist[u] + weight < dist[v]:
                return None, None

        path = []
        current_node = target_node
        while current_node is not None:
            path.insert(0, current_node)
            current_node = prev[current_node]

        return path, self.calculate_path_cost(graph, path)

    def dfs(self, graph, start, goal):
        visited = set()
        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()
            if current == goal:
                return path, self.calculate_path_cost(graph, path)

            visited.add(current)

            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None, None

    def find_nearest_edge(self, graph, point):
        nearest_edge = None
        min_distance = float("inf")

        for edge in graph.edges():
            u, v = edge
            projection_point = self.calculate_projection(point, u, v)
            distance = self.calculate_distance(point, projection_point)
            if distance < min_distance:
                min_distance = distance
                nearest_edge = edge
                p = projection_point

        return nearest_edge[0], nearest_edge[1], p

    def find_shortest_path(self):
        if len(self.marker_list) < 2:
            messagebox.showinfo("Error", "Must choose start point and end point!")
            return

        start_time = time.time()
        self.map_widget.delete_all_path()
        self.make_graph()
        start = self.marker_list[0].position
        goal = self.marker_list[1].position

        (
            nearest_edge_start_1,
            nearest_edge_start_2,
            projection_start,
        ) = self.find_nearest_edge(self.graph, start)
        (
            nearest_edge_goal_1,
            nearest_edge_goal_2,
            projection_goal,
        ) = self.find_nearest_edge(self.graph, goal)
        nearest_edge_start = (nearest_edge_start_1, nearest_edge_start_2)
        nearest_edge_goal = (nearest_edge_goal_1, nearest_edge_goal_2)

        self.draw_dash_line([start, projection_start])
        self.draw_dash_line([goal, projection_goal])

        start, goal = projection_start, projection_goal
        print(
            f"Diem xuat phat la: {start}\nCanh gan nhat diem xuat phat la: {nearest_edge_start}\nDiem dich la: {goal}\nCanh gan nhat diem dich la: {nearest_edge_goal}"
        )
        self.graph.add_nodes_from([start, goal])

        self.graph.add_edge(
            nearest_edge_start_1,
            start,
            weight=self.calculate_distance(nearest_edge_start_1, start),
        )
        self.graph.add_edge(
            start,
            nearest_edge_start_2,
            weight=self.calculate_distance(nearest_edge_start_2, start),
        )

        self.graph.add_edge(
            nearest_edge_goal_1,
            goal,
            weight=self.calculate_distance(nearest_edge_goal_1, goal),
        )
        self.graph.add_edge(
            goal,
            nearest_edge_goal_2,
            weight=self.calculate_distance(goal, nearest_edge_goal_2),
        )
        self.graph.remove_edge(nearest_edge_goal_1, nearest_edge_goal_2)
        if nearest_edge_start != nearest_edge_goal:
            self.graph.remove_edge(nearest_edge_start_1, nearest_edge_start_2)

        if (nearest_edge_start_2, nearest_edge_start_1) in self.graph.edges:
            self.graph.add_edge(
                nearest_edge_start_2,
                start,
                weight=self.calculate_distance(nearest_edge_start_2, start),
            )
            self.graph.add_edge(
                start,
                nearest_edge_start_1,
                weight=self.calculate_distance(nearest_edge_start_1, start),
            )
            self.graph.remove_edge(nearest_edge_start_2, nearest_edge_start_1)

        if (nearest_edge_goal_2, nearest_edge_goal_1) in self.graph.edges:
            self.graph.add_edge(
                nearest_edge_goal_2,
                goal,
                weight=self.calculate_distance(nearest_edge_goal_2, goal),
            )
            self.graph.add_edge(
                goal,
                nearest_edge_goal_1,
                weight=self.calculate_distance(goal, nearest_edge_goal_1),
            )
            self.graph.remove_edge(nearest_edge_goal_2, nearest_edge_goal_1)

        if nearest_edge_start == nearest_edge_goal:
            print("Canh dau va cuoi giong nhau")
            d1 = self.calculate_distance(projection_start, nearest_edge_start_1)
            d2 = self.calculate_distance(projection_goal, nearest_edge_start_1)
            if d1 < d2:
                print("Start nam gan dau hon goal")
                self.graph.remove_edges_from(
                    [
                        (start, nearest_edge_start_2),
                        (nearest_edge_start_1, goal),
                    ]
                )
                self.graph.add_edge(
                    start, goal, weight=self.calculate_distance(start, goal)
                )
                if (nearest_edge_start_2, start) in self.graph.edges:
                    print("Canh 2 chieu")
                    self.graph.remove_edges_from(
                        [(nearest_edge_start_2, start), (goal, nearest_edge_start_1)]
                    )
                    self.graph.add_edge(
                        goal, start, weight=self.calculate_distance(start, goal)
                    )

            else:
                print("Goal nam gan dau hon start")
                self.graph.remove_edges_from(
                    [
                        (nearest_edge_start_1, start),
                        (goal, nearest_edge_start_2),
                    ]
                )
                self.graph.add_edge(
                    goal, start, weight=self.calculate_distance(start, goal)
                )
                if (start, nearest_edge_start_1) in self.graph.edges:
                    print("Canh 2 chieu")
                    self.graph.remove_edges_from(
                        [(start, nearest_edge_start_1), (nearest_edge_start_2, goal)]
                    )
                    self.graph.add_edge(
                        start, goal, weight=self.calculate_distance(start, goal)
                    )

        if algorithm == "A*":
            path, cost = self.a_star(self.graph, start, goal)
        elif algorithm == "BFS":
            path, cost = self.bfs(self.graph, start, goal)
        elif algorithm == "Dijkstra":
            path, cost = self.dijkstra(self.graph, start, goal)
        elif algorithm == "Bellman-Ford":
            path, cost = self.bellman_ford(self.graph, start, goal)
        elif algorithm == "DFS":
            path, cost = self.dfs(self.graph, start, goal)
        else:
            messagebox.showerror("Error", "Invalid algorithm")
            return

        if path:
            end_time = time.time()
            running_time = end_time - start_time
            messageCost = f"Thoi gian tim kiem cua thuat toan {algorithm} la: {running_time}s. Tong quang duong di: {cost}"
            messagebox.showinfo("Path found", "Shortest path found!\n" + messageCost)
            self.display_map(path)
        else:
            messagebox.showinfo("Path not found", "No path found!")

    # =================Các hàm xử lý sự kiện button==================

    def handle_a_star(self):
        global algorithm
        algorithm = "A*"
        self.find_shortest_path()

    def handle_bfs(self):
        global algorithm
        algorithm = "BFS"
        self.find_shortest_path()

    def handle_dijkstra(self):
        global algorithm
        algorithm = "Dijkstra"
        self.find_shortest_path()

    def handle_bellman_ford(self):
        global algorithm
        algorithm = "Bellman-Ford"
        self.find_shortest_path()

    def handle_dfs(self):
        global algorithm
        algorithm = "DFS"
        self.find_shortest_path()

    # =====================Các hàm xử lý giao diện==========================

    def search_event(self, event=None):
        self.map_widget.set_address(self.entry.get())

    def clear_marker_event(self):
        self.marker_list.clear()
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()

    def change_appearance_mode(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_map(self, new_map: str):
        if new_map == "OpenStreetMap":
            self.map_widget.set_tile_server(
                "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
            )
        elif new_map == "Google normal":
            self.map_widget.set_tile_server(
                "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga",
                max_zoom=22,
            )
        elif new_map == "Google satellite":
            self.map_widget.set_tile_server(
                "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga",
                max_zoom=22,
            )

    def change_test(self, new_test: str):
        if new_test == "Find Shortest Path By A*":
            self.handle_a_star()
        elif new_test == "Find Shortest Path By BFS":
            self.handle_bfs()
        elif new_test == "Find Shortest Path By Dijkstra":
            self.handle_dijkstra()
        elif new_test == "Find Shortest Path By Bellman-Ford":
            self.handle_bellman_ford()
        elif new_test == "Find Shortest Path By DFS":
            self.handle_dfs()

    def add_marker_event(self, coords):
        if len(self.marker_list) == 2 or len(self.marker_list) == 0:
            self.clear_marker_event()
            print("Start:", coords)
            new_marker = self.map_widget.set_marker(coords[0], coords[1], text="Start")
        else:
            print("Goal:", coords)
            new_marker = self.map_widget.set_marker(coords[0], coords[1], text="End")

        self.marker_list.append(new_marker)

    def display_map(self, path):
        for i in range(len(path) - 1):
            print(path[i], " -> ")
            self.map_widget.set_path([path[i], path[i + 1]])
        print(path[len(path) - 1])

    def draw_dash_line(self, line):
        start, end = line[0], line[1]
        x1, y1 = start[0], start[1]
        x2, y2 = end[0], end[1]
        d1 = (x2 - x1)/10
        d2 = (y2 - y1)/10
        
        for i in range(10):
            if i % 2 == 0:
                self.map_widget.set_path([(x1, y1), (x1 + d1, y1 + d2)])
    
            x1 = x1 + d1
            y1 = y1 + d2

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
