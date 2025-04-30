import os
import json
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ConstructSurfaceModifier,
    InvertSelectionModifier,
    DeleteSelectedModifier,
    ClusterAnalysisModifier
)

class SurfaceMeshProcessor:
   
    def __init__(self, defect_path, radius, smoothing_level,cutoff, out_base="out"):
        self.defect_path = defect_path
        self.radius = radius
        self.cutoff = cutoff
        self.smoothing_level = smoothing_level
        self.out_base = out_base
        self.dump_dir = os.path.join(self.out_base, "dump")

    def create_output_dirs(self):
        for sub in ("dump", "json", "csv"):
            path = os.path.join(self.out_base, sub)
            os.makedirs(path, exist_ok=True)
        print(f"Directorios creados en '{self.out_base}': dump, json, csv")

    def process(self):
        
        # Asegurar directorios
        self.create_output_dirs()

        # Cargar pipeline
        pipeline = import_file(self.defect_path)
        pipeline.modifiers.append(
            ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level,
                select_surface_particles=True
            )
        )
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff,unwrap_particles=True))

      

        out_file = os.path.join(self.dump_dir, "key_areas.dump")
        try:
            export_file(
                pipeline,
                out_file,
                "lammps/dump",
                columns=[
                    "Particle Identifier",
                    "Particle Type",
                    "Position.X",
                    "Position.Y",
                    "Position.Z"
                ]
            )
            print(f"[OK] Exportado en: {out_file}")
        except Exception as e:
            print(f"[ERROR] export_file() falló: {e}")


import os
import pandas as pd
from ovito.io import import_file
from ovito.modifiers import ClusterAnalysisModifier

class DumpToCsv:
    """
    Convierte un archivo LAMMPS dump en un CSV con columnas x, y, z y cluster.
    """
    def __init__(self, dump_path, csv_dir="out/csv", cluster_cutoff=1.0):
        self.dump_path = dump_path
        self.csv_dir = csv_dir
        self.cluster_cutoff = cluster_cutoff

    def convert(self):
        # 1) Asegurar directorio de salida
        os.makedirs(self.csv_dir, exist_ok=True)

        # 2) Importar dump y analizar clusters
        pipeline = import_file(self.dump_path)
        pipeline.modifiers.append(
            ClusterAnalysisModifier(cutoff=self.cluster_cutoff)
        )
        data = pipeline.compute()

        # 3) Extraer posiciones y cluster IDs
        positions = data.particles.positions   # array (N,3)
        cluster_ids = data.particles['Cluster'].array  # property añadida

        df = pd.DataFrame({
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
            'cluster': cluster_ids
        })

        base = os.path.splitext(os.path.basename(self.dump_path))[0]
        out_file = os.path.join(self.csv_dir, f"{base}.csv")
        df.to_csv(out_file, index=False)
        print(f"[OK] CSV exportado a: {out_file}")
        return out_file
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class KMeansReclusterCSV:
    def __init__(self, input_csv, n_clusters, output_csv=None, random_state=0):
        self.input_csv = input_csv
        self.n_clusters = n_clusters
        self.random_state = random_state
        base, _ = os.path.splitext(input_csv)
        self.output_csv = output_csv or f"{base}_reclustered.csv"
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.input_csv)

    def apply_kmeans(self):
        coords = self.df[['x','y','z']].values
        kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", random_state=self.random_state).fit(coords)
        self.df['cluster'] = kmeans.labels_ + 1

    def save(self):
        os.makedirs(os.path.dirname(self.output_csv) or '.', exist_ok=True)
        self.df.to_csv(self.output_csv, index=False)

    def run(self):
        self.load()
        self.apply_kmeans()
        self.save()

    def export_by_cluster(self, output_dir=None):
        base, _ = os.path.splitext(self.input_csv)
        output_dir = output_dir or f"{base}_clusters"
        os.makedirs(output_dir, exist_ok=True)
        for c, group in self.df.groupby('cluster'):
            out_path = os.path.join(output_dir, f"cluster_{c}.csv")
            group.to_csv(out_path, index=False)

    def compute_clusters_dispersion(self, clusters_dir=None):
        base, _ = os.path.splitext(self.input_csv)
        clusters_dir = clusters_dir or f"{base}_clusters"
        results = {}
        for fname in os.listdir(clusters_dir):
            if fname.startswith('cluster_') and fname.endswith('.csv'):
                path = os.path.join(clusters_dir, fname)
                df = pd.read_csv(path)
                coords = df[['x','y','z']].values
                center = coords.mean(axis=0)
                dists = np.linalg.norm(coords - center, axis=1)
                dispersion = dists.mean()
                cluster_id = int(fname.split('_')[1].split('.')[0])
                results[cluster_id] = {
                    'center_of_mass': center.tolist(),
                    'dispersion': float(dispersion)
                }
        return results

if __name__ == "__main__":
   
    with open("input_params.json", "r") as f:
        cfg = json.load(f)["CONFIG"][0]
    processor = SurfaceMeshProcessor(
        defect_path=cfg["defect"],
        radius=cfg["radius"],
        smoothing_level=cfg["smoothing_level"],
        cutoff=cfg["cutoff"],
        out_base="out"
    )
    processor.process()

    
    converter = DumpToCsv("out/dump/key_areas.dump", csv_dir="out/csv", cluster_cutoff=1.5)
    converter.convert()

   
    processor = KMeansReclusterCSV("out/csv/key_areas.csv", n_clusters=5)
    processor.run()
    processor.export_by_cluster()
    result =processor.compute_clusters_dispersion()
    print(result)




