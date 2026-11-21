import csv
from datetime import datetime
import heapq
import math
import os
import time
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
# Configuració necessària perquè la llibreria gràfica no doni problemes a Mac/Linux
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

import osmnx as ox

# Zones on descarregarem el mapa per l'estudi
POBLACIONS = ["Viladecans, Barcelona, Spain", "Gavà, Barcelona, Spain"]
FITXER_GRAF = "mapa_viladecans_gava_estudi.graphml"

# Paràmetres que utilitzarem per a la simulació i animació
W_ASTAR = 2.0
PAS_NORMAL = 15
PAS_RAPID = 40
LLINDAR_RAPID = 800

# Rutes per defecte per fer proves ràpides sense haver de fer clic
RUTES = [
    ("Ruta Urbana (Miramar -> Renfe)", (41.3175, 2.0057), (41.3112, 2.0298)),
    ("Ruta Urbana Mitjana (Torre-roja -> Vilamarina)", (41.3250, 2.0110), (41.3090, 2.0310)),
    ("Ruta Interurbana (Gavà -> Viladecans)", (41.3070, 2.0020), (41.3155, 2.0180)),
    ("Ruta Llarga (Centre Gavà -> Centre Viladecans)", (41.3050, 1.9950), (41.3180, 2.0250)),
]

def distancia_haversine(G, n1, n2):
    # Fórmula de Haversine
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]

    R = 6371000.0
    lat1, lat2 = math.radians(y1), math.radians(y2)
    dlat = math.radians(y2 - y1)
    dlon = math.radians(x2 - x1)

    a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
    if a > 1.0: 
        a = 1.0
    if a < 0.0: 
        a = 0.0
    return 2.0 * R * math.asin(math.sqrt(a))


def pilla_veins(G, node):
    veins = []
    for vei, dades_aresta in G.adj[node].items():
        if G.is_multigraph():
            # A vegades hi ha varis carrers paral·lels, ens quedem el més curt
            pes = min(float(d.get("length", d.get("weight", 1.0))) for d in dades_aresta.values())
        else:
            pes = float(dades_aresta.get("length", dades_aresta.get("weight", 1.0)))
        veins.append((vei, pes))
    return veins


def reconstruir_ruta(predecessors, origen, desti):
    if desti not in predecessors and origen != desti:
        return None
        
    ruta = []
    actual = desti
    while actual in predecessors:
        ruta.append(actual)
        actual = predecessors[actual]
        
    ruta.append(origen)
    ruta.reverse()
    return ruta

def executar_dijkstra(G, origen, desti):
    # Fem el recorregut amb Dijkstra clàssic
    t0 = time.perf_counter()
    cua = [(0.0, origen)]
    distancies = {origen: 0.0}
    predecessors = {}
    visitats = set() 
    explorats_ordre = []

    while cua:
        cost_actual, actual = heapq.heappop(cua)
        
        # Evitem processar el node més d'una vegada
        if actual in visitats:
            continue
            
        visitats.add(actual)
        explorats_ordre.append(actual)

        # Si arribem al node final, ja podem parar
        if actual == desti:
            break

        for vei, pes in pilla_veins(G, actual):
            nou_cost = cost_actual + pes
            if vei not in distancies or nou_cost < distancies[vei]:
                distancies[vei] = nou_cost
                predecessors[vei] = actual
                heapq.heappush(cua, (nou_cost, vei))

    tf = time.perf_counter()
    temps_total = tf - t0
    cami = reconstruir_ruta(predecessors, origen, desti)
    return cami, distancies.get(desti, float('inf')), explorats_ordre, temps_total


def executar_astar(G, origen, desti, pes_h=1.0):
    t0 = time.perf_counter()
    cua = [(0.0, 0.0, origen)]
    costos_g = {origen: 0.0}
    predecessors = {}
    visitats = set()
    explorats_ordre = []

    while cua:
        cost_f, cost_g_actual, actual = heapq.heappop(cua)
        
        if actual in visitats:
            continue
            
        visitats.add(actual)
        explorats_ordre.append(actual)

        if actual == desti:
            break

        for vei, pes in pilla_veins(G, actual):
            nou_g = cost_g_actual + pes
            if vei not in costos_g or nou_g < costos_g[vei]:
                costos_g[vei] = nou_g
                predecessors[vei] = actual
                
                h = distancia_haversine(G, vei, desti)
                nou_f = nou_g + (pes_h * h)
                heapq.heappush(cua, (nou_f, nou_g, vei))

    tf = time.perf_counter()
    temps_total = tf - t0
    cami = reconstruir_ruta(predecessors, origen, desti)
    return cami, costos_g.get(desti, float('inf')), explorats_ordre, temps_total

def carregar_mapa():
    # Comprovem si ja tenim el fitxer local per estalviar temps
    if os.path.exists(FITXER_GRAF):
        return ox.load_graphml(FITXER_GRAF)
    
    print("Descarregant mapa d'internet, aixo pot trigar una mica...")
    G = ox.graph_from_place(POBLACIONS, network_type="drive", simplify=True)
    ox.save_graphml(G, FITXER_GRAF)
    return G


class AppSimulador:
    def __init__(self, finestra_principal, G):
        self.G = G
        self.finestra = finestra_principal
        self.node_origen = None
        self.node_desti = None
        self.animacio_activa = False
        self.dades_historial = []

        self.finestra.title("Simulador Rutes")
        self.finestra.geometry("1450x850")
        self.finestra.configure(bg="#111111")

        self.nodes_id = list(self.G.nodes())
        self.geometria_carrers = self.extreure_geometria()

        self.crear_interficie()
        self.reiniciar_mapa()

    def extreure_geometria(self):
        geom = []
        for u, v, k, d in self.G.edges(keys=True, data=True):
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                geom.append(list(zip(xs, ys)))
            else:
                if 'x' in self.G.nodes[u] and 'x' in self.G.nodes[v]:
                    p1 = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                    p2 = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                    geom.append([p1, p2])
        return geom

    def buscar_node_proper(self, x, y):
        dist_min = float('inf')
        node_trobat = None
        for n in self.nodes_id:
            nx_pos = self.G.nodes[n]['x']
            ny_pos = self.G.nodes[n]['y']
            d = math.hypot(nx_pos - x, ny_pos - y)
            if d < dist_min:
                dist_min = d
                node_trobat = n
        return node_trobat

    def crear_interficie(self):
        # Configuració de la finestra
        panell = tk.Frame(self.finestra, width=380, bg="#1a1a1c", padx=20, pady=20)
        panell.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(panell, text="CONTROLS", font=("Segoe UI", 12, "bold"), bg="#1a1a1c", fg="#ffffff").pack(pady=(0, 15))

        # Selector de ruta
        self.var_ruta = tk.StringVar(value=RUTES[0][0])
        cb = ttk.Combobox(panell, textvariable=self.var_ruta, values=[r[0] for r in RUTES], state="readonly")
        cb.pack(fill=tk.X, pady=(4, 15))
        cb.bind("<<ComboboxSelected>>", self.canvi_de_ruta)
        
        # Algoritmes
        tk.Button(panell, text="Dijkstra", font=("Segoe UI", 9, "bold"), bg="#1e88e5", fg="white", relief=tk.FLAT, command=lambda: self.executar_simulacio("dijkstra")).pack(fill=tk.X, pady=2, ipady=4)
        tk.Button(panell, text="A* Normal", font=("Segoe UI", 9, "bold"), bg="#fbc02d", fg="black", relief=tk.FLAT, command=lambda: self.executar_simulacio("astar")).pack(fill=tk.X, pady=2, ipady=4)
        tk.Button(panell, text=f"A* Ponderat (W={W_ASTAR})", font=("Segoe UI", 9, "bold"), bg="#e91e63", fg="white", relief=tk.FLAT, command=lambda: self.executar_simulacio("wastar")).pack(fill=tk.X, pady=2, ipady=4)
        tk.Button(panell, text="Tots", font=("Segoe UI", 9, "bold"), bg="#8e24aa", fg="white", relief=tk.FLAT, command=lambda: self.executar_simulacio("tots")).pack(fill=tk.X, pady=(8, 15), ipady=5)

        tk.Button(panell, text="Netejar mapa", font=("Segoe UI", 9), bg="#444444", fg="white", relief=tk.FLAT, command=self.netejar_tot).pack(fill=tk.X, pady=(0, 15))

        # Exportació
        tk.Button(panell, text="Exportar CSV", font=("Segoe UI", 9), bg="#2e7d32", fg="white", relief=tk.FLAT, command=self.exportar_dades_csv).pack(fill=tk.X, pady=2)
        tk.Button(panell, text="Guardar PNG", font=("Segoe UI", 9), bg="#00838f", fg="white", relief=tk.FLAT, command=self.guardar_foto).pack(fill=tk.X, pady=2)

        # Consola
        self.log = tk.Text(panell, height=14, font=("Consolas", 8), bg="#111811", fg="#a5d6a7", relief=tk.FLAT, padx=10, pady=10)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        self.log.insert(tk.END, "Iniciat correctament.\n")

        st = ttk.Style()
        st.theme_use('default')
        st.configure('TNotebook', background='#111111', borderwidth=0)
        st.configure('TNotebook.Tab', background='#1a1a1c', foreground='white', padding=[12, 4], font=('Segoe UI', 9))
        st.map('TNotebook.Tab', background=[('selected', '#1e88e5')])

        self.tabs = ttk.Notebook(self.finestra)
        self.tabs.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_mapa = tk.Frame(self.tabs, bg="#111111")
        self.tabs.add(self.tab_mapa, text="Mapa")

        self.fig_mapa = plt.Figure(figsize=(8, 8), facecolor="#111111")
        self.ax_mapa = self.fig_mapa.add_subplot(111)
        self.fig_mapa.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas_mapa = FigureCanvasTkAgg(self.fig_mapa, master=self.tab_mapa)
        self.canvas_mapa.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig_mapa.canvas.mpl_connect('button_press_event', self.gestiona_clic_mapa)

        self.tab_stats = tk.Frame(self.tabs, bg="#111111")
        self.tabs.add(self.tab_stats, text="Estadístiques")

        self.fig_stats = plt.Figure(figsize=(7, 5), facecolor="#111111")
        self.ax_stats = self.fig_stats.add_subplot(111)
        self.ax_stats.set_facecolor("#111111")
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=self.tab_stats)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.finestra.after(200, self.canvi_de_ruta)

    def reiniciar_mapa(self):
        if self.animacio_activa:
            return
        
        self.ax_mapa.clear()
        self.ax_mapa.set_facecolor("#111111")

        lc = LineCollection(self.geometria_carrers, colors="#636e72", linewidths=1.0, alpha=0.7, zorder=1)
        self.ax_mapa.add_collection(lc)

        self.sc_d = self.ax_mapa.scatter([], [], s=14, color="#1e88e5", alpha=0.9, zorder=2, label="Dijkstra")
        self.sc_a = self.ax_mapa.scatter([], [], s=14, color="#fbc02d", alpha=0.9, zorder=3, label="A* Normal")
        self.sc_wa = self.ax_mapa.scatter([], [], s=14, color="#e91e63", alpha=1.0, zorder=4, label="A* Ponderat")

        self.linia_cami, = self.ax_mapa.plot([], [], c="#ffffff", lw=2.8, zorder=5, label="Cami")
        self.pnts_extrems = self.ax_mapa.scatter([], [], s=90, c=[], marker="o", edgecolors="white", linewidths=1.5, zorder=6)

        xs = [self.G.nodes[n]['x'] for n in self.nodes_id]
        ys = [self.G.nodes[n]['y'] for n in self.nodes_id]
        if xs:
            self.ax_mapa.set_xlim(min(xs), max(xs))
            self.ax_mapa.set_ylim(min(ys), max(ys))

        self.ax_mapa.axis("off")
        self.ax_mapa.legend(loc="upper left", facecolor="#1a1a1c", edgecolor="#444444", labelcolor="white", fontsize=8)
        self.canvas_mapa.draw()
        self.actualitzar_marcadors()

    def actualitzar_marcadors(self):
        x_pts = []
        y_pts = []
        colors = []
        
        if self.node_origen is not None:
            x_pts.append(self.G.nodes[self.node_origen]['x'])
            y_pts.append(self.G.nodes[self.node_origen]['y'])
            colors.append("#4caf50")
            
        if self.node_desti is not None:
            x_pts.append(self.G.nodes[self.node_desti]['x'])
            y_pts.append(self.G.nodes[self.node_desti]['y'])
            colors.append("#f44336")

        if x_pts:
            self.pnts_extrems.set_offsets(np.column_stack((x_pts, y_pts)))
            self.pnts_extrems.set_color(colors)
            self.canvas_mapa.draw_idle()

    def canvi_de_ruta(self, event=None):
        nom = self.var_ruta.get()
        for r in RUTES:
            if r[0] == nom:
                self.node_origen = self.buscar_node_proper(r[1][1], r[1][0])
                self.node_desti = self.buscar_node_proper(r[2][1], r[2][0])

                self.reiniciar_mapa()
                self.log.insert(tk.END, f"Carregat: {nom.split('(')[0].strip()}\n")
                self.log.see(tk.END)
                break

    def netejar_tot(self):
        self.node_origen = None
        self.node_desti = None
        self.reiniciar_mapa()
        self.log.insert(tk.END, "Netejat.\n")
        self.log.see(tk.END)

    def gestiona_clic_mapa(self, event):
        if event.inaxes is None or self.animacio_activa:
            return

        clic_n = self.buscar_node_proper(event.xdata, event.ydata)

        if self.node_origen is None:
            self.node_origen = clic_n
        elif self.node_desti is None:
            self.node_desti = clic_n
        else:
            # Si ja estan establerts, canviem el que quedi més a prop d'on hem fet el clic
            xo = self.G.nodes[self.node_origen]['x']
            yo = self.G.nodes[self.node_origen]['y']
            xd = self.G.nodes[self.node_desti]['x']
            yd = self.G.nodes[self.node_desti]['y']

            dist_orig = math.hypot(xo - event.xdata, yo - event.ydata)
            dist_dest = math.hypot(xd - event.xdata, yd - event.ydata)

            self.reiniciar_mapa()
            if dist_orig < dist_dest:
                self.node_origen = clic_n
            else:
                self.node_desti = clic_n

        self.var_ruta.set("Ruta Personalitzada")
        self.actualitzar_marcadors()

    def executar_simulacio(self, mode):
        if self.animacio_activa or self.node_origen is None or self.node_desti is None:
            return

        self.tabs.select(self.tab_mapa)
        self.reiniciar_mapa()
        self.log.insert(tk.END, f"Calculant {mode}...\n")
        self.finestra.update()

        self.res_d = None
        self.res_a = None
        self.res_wa = None

        nom_ruta_actual = self.var_ruta.get().split("(")[0].strip()

        # Calculem les rutes i guardem els resultats per pintar-los després
        if mode in ["dijkstra", "tots"]:
            cami, dist, nodes, temps = executar_dijkstra(self.G, self.node_origen, self.node_desti)
            if cami:
                self.res_d = {"cami": cami, "dist": dist, "nodes": nodes, "temps": temps}
                self.dades_historial.append([nom_ruta_actual, "Dijkstra", round(dist, 1), len(nodes), round(temps, 4)])

        if mode in ["astar", "tots"]:
            cami, dist, nodes, temps = executar_astar(self.G, self.node_origen, self.node_desti, pes_h=1.0)
            if cami:
                self.res_a = {"cami": cami, "dist": dist, "nodes": nodes, "temps": temps}
                self.dades_historial.append([nom_ruta_actual, "A_Star", round(dist, 1), len(nodes), round(temps, 4)])

        if mode in ["wastar", "tots"]:
            cami, dist, nodes, temps = executar_astar(self.G, self.node_origen, self.node_desti, pes_h=W_ASTAR)
            if cami:
                self.res_wa = {"cami": cami, "dist": dist, "nodes": nodes, "temps": temps}
                self.dades_historial.append([nom_ruta_actual, "A_Star_Ponderat", round(dist, 1), len(nodes), round(temps, 4)])

        # Mirem si tenim alguna ruta
        if not any([self.res_d, self.res_a, self.res_wa]):
            self.log.insert(tk.END, "Error, camí no trobat.\n")
            self.log.see(tk.END)
            return

        self.mode_actiu = mode
        self.idx_d = 0
        self.idx_a = 0
        self.idx_wa = 0
        self.animacio_activa = True

        max_n = 0
        if self.res_d:
            max_n = max(max_n, len(self.res_d["nodes"]))
        if self.res_a:
            max_n = max(max_n, len(self.res_a["nodes"]))
        if self.res_wa:
            max_n = max(max_n, len(self.res_wa["nodes"]))

        # Accelerem l'animació si hi ha molts punts a recórrer
        self.pas_actual = PAS_RAPID if max_n > LLINDAR_RAPID else PAS_NORMAL

        self.dibuixar_frame_animacio()

    def dibuixar_frame_animacio(self):
        encara_dibuixant = False

        if self.res_d and self.idx_d < len(self.res_d["nodes"]):
            self.idx_d = min(self.idx_d + self.pas_actual, len(self.res_d["nodes"]))
            pts = []
            for n in self.res_d["nodes"][:self.idx_d]:
                pts.append([self.G.nodes[n]['x'], self.G.nodes[n]['y']])
            if pts:
                self.sc_d.set_offsets(np.array(pts))
            encara_dibuixant = True

        if self.res_a and self.idx_a < len(self.res_a["nodes"]):
            self.idx_a = min(self.idx_a + self.pas_actual, len(self.res_a["nodes"]))
            pts = []
            for n in self.res_a["nodes"][:self.idx_a]:
                pts.append([self.G.nodes[n]['x'], self.G.nodes[n]['y']])
            if pts:
                self.sc_a.set_offsets(np.array(pts))
            encara_dibuixant = True

        if self.res_wa and self.idx_wa < len(self.res_wa["nodes"]):
            self.idx_wa = min(self.idx_wa + self.pas_actual, len(self.res_wa["nodes"]))
            pts = []
            for n in self.res_wa["nodes"][:self.idx_wa]:
                pts.append([self.G.nodes[n]['x'], self.G.nodes[n]['y']])
            if pts:
                self.sc_wa.set_offsets(np.array(pts))
            encara_dibuixant = True

        self.canvas_mapa.draw_idle()

        if encara_dibuixant:
            self.finestra.after(15, self.dibuixar_frame_animacio)
        else:
            self.animacio_activa = False
            self.tancar_simulacio()

    def tancar_simulacio(self):
        res = None
        if self.mode_actiu == "wastar" and self.res_wa:
            res = self.res_wa
        elif self.mode_actiu == "astar" and self.res_a:
            res = self.res_a
        elif self.res_d:
            res = self.res_d
        elif self.res_a:
            res = self.res_a
        elif self.res_wa:
            res = self.res_wa

        if res and res["cami"]:
            xs = []
            ys = []
            for n in res["cami"]:
                xs.append(self.G.nodes[n]['x'])
                ys.append(self.G.nodes[n]['y'])
            self.linia_cami.set_data(xs, ys)

        self.canvas_mapa.draw_idle()

        vd = len(self.res_d["nodes"]) if self.res_d else 0
        va = len(self.res_a["nodes"]) if self.res_a else 0
        vwa = len(self.res_wa["nodes"]) if self.res_wa else 0
        self.dibuixar_barres_estadistiques(vd, va, vwa)

        self.log.insert(tk.END, "--- FI ---\n")
        if self.res_d:
            self.log.insert(tk.END, f"Dijkstra: {self.res_d['temps']:.4f}s | {len(self.res_d['nodes'])} nodes\n")
        if self.res_a:
            self.log.insert(tk.END, f"A*: {self.res_a['temps']:.4f}s | {len(self.res_a['nodes'])} nodes\n")
        if self.res_wa:
            self.log.insert(tk.END, f"A* Pond: {self.res_wa['temps']:.4f}s | {len(self.res_wa['nodes'])} nodes\n")
        self.log.insert(tk.END, "\n")
        self.log.see(tk.END)

    def dibuixar_barres_estadistiques(self, num_dijkstra, num_astar, num_wastar):
        self.ax_stats.clear()
        self.ax_stats.set_facecolor("#111111")

        categories = ['Dijkstra', 'A*', 'A* Ponderat']
        valors = [num_dijkstra, num_astar, num_wastar]
        colors = ['#1e88e5', '#fbc02d', '#e91e63']

        bars = self.ax_stats.bar(categories, valors, color=colors, width=0.45)
        self.ax_stats.tick_params(colors='white')
        self.ax_stats.spines['bottom'].set_color('white')
        self.ax_stats.spines['left'].set_color('white')
        self.ax_stats.spines['top'].set_visible(False)
        self.ax_stats.spines['right'].set_visible(False)

        for b in bars:
            altura = b.get_height()
            if altura > 0:
                self.ax_stats.text(b.get_x() + b.get_width() / 2, altura + (max(valors) * 0.02),
                                   f"{int(altura)}", ha='center', va='bottom', color='white', fontsize=9)

        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

    def exportar_dades_csv(self):
        if not self.dades_historial:
            return

        nom_fitxer = f"Dades_{datetime.now().strftime('%H%M%S')}.csv"
        try:
            with open(nom_fitxer, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["Ruta", "Algo", "Dist", "Nodes", "Temps"])
                for fila in self.dades_historial:
                    writer.writerow(fila)
            self.log.insert(tk.END, f"CSV guardat -> {nom_fitxer}\n")
        except OSError as err:
            self.log.insert(tk.END, f"Error al guardar CSV: {err}\n")
        self.log.see(tk.END)

    def guardar_foto(self):
        nom_foto = f"Mapa_{datetime.now().strftime('%H%M%S')}.png"
        try:
            self.fig_mapa.savefig(nom_foto, dpi=300, bbox_inches='tight', facecolor='#111111')
            self.log.insert(tk.END, f"Imatge guardada -> {nom_foto}\n")
        except OSError as err:
            self.log.insert(tk.END, f"Error guardant foto: {err}\n")
        self.log.see(tk.END)


if __name__ == "__main__":
    graf = carregar_mapa()
    if graf is not None:
        root = tk.Tk()
        app = AppSimulador(root, graf)
        root.mainloop()