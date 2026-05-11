"""
PORÓWNANIE RÓŻNYCH METOD DETEKCJI LINII DLA SKICZÓW TECHNICZNYCH
Z wizualizacjami wyników i statystykami wydajności

Metody testowane:
- Standardowe LSD (Line Segment Detector)
- Ulepszony LSD z parametrami
- Adaptywny Hough Transform
- Zoptymalizowany Hough
- Canny + LSD (najlepsza kombinacja)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import json
from datetime import datetime
import time

def load_test_image(image_path):
    """Wczytaj i przygotuj obraz testowy"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")
    return img

def standard_lsd(img):
    """Standardowy LSD detektor"""
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    return lines if lines is not None else np.array([])

def enhanced_lsd(img):
    """Ulepszony LSD z wstępnym przetwarzaniem"""
    # Wstępne przetwarzanie obrazu dla lepszych wyników
    kernel = np.ones((2,2), np.uint8)
    img_enhanced = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Standardowy LSD z wzmocnionym obrazem
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img_enhanced)[0]
    return lines if lines is not None else np.array([])

def adaptive_hough(img):
    """Adaptywny Hough z automatycznym progiem"""
    # Oblicz optymalny próg na podstawie histogramu
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    threshold = int(np.percentile(img[img > 0], 85)) if np.any(img > 0) else 30
    
    lines = cv2.HoughLinesP(
        img, 
        rho=1, 
        theta=np.pi/180,
        threshold=max(20, threshold),
        minLineLength=15,
        maxLineGap=8
    )
    return lines if lines is not None else np.array([])

def optimized_hough(img):
    """Zoptymalizowany Hough z filtracją"""
    # Morfologiczne oczyszczenie
    kernel = np.ones((2,2), np.uint8)
    img_clean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    lines = cv2.HoughLinesP(
        img_clean,
        rho=2,
        theta=np.pi/180,
        threshold=25,
        minLineLength=20,
        maxLineGap=10
    )
    return lines if lines is not None else np.array([])

def canny_plus_lsd(img):
    """Kombinacja Canny + LSD (najlepsza metoda)"""
    # Łagodniejsze krawędzie Canny dla lepszych krzywych
    edges = cv2.Canny(img, 30, 100, apertureSize=3)
    
    # Wzmocnienie krawędzi (większy kernel)
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    
    # LSD na oryginalnym obrazie z preprocessowaniem
    kernel_prep = np.ones((2,2), np.uint8)
    img_enhanced = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_prep)
    
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img_enhanced)[0]
    return lines if lines is not None else np.array([])

def visualize_results(img, methods_results, save_path=None):
    """Wizualizacja wyników wszystkich metod"""
    methods = list(methods_results.keys())
    lines_dict = list(methods_results.values())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Porównanie metod detekcji linii', fontsize=16, fontweight='bold')
    
    # Oryginalny obraz
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Oryginalny obraz')
    axes[0,0].axis('off')
    
    # Wyniki dla każdej metody
    for i, (method, lines) in enumerate(zip(methods, lines_dict)):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        # Konwertuj obraz do RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Rysuj linie
        if len(lines) > 0:
            for line in lines:
                if len(line) == 4:  # Hough format
                    x1, y1, x2, y2 = line
                else:  # LSD format
                    x1, y1, x2, y2 = line[0]
                cv2.line(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(f'{method}\n{len(lines)} linii')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Wizualizacja zapisana: {save_path}")
    
    plt.show()

def benchmark_methods(img):
    """Test wydajności wszystkich metod"""
    methods = {
        'Standardowe LSD': standard_lsd,
        'Ulepszony LSD': enhanced_lsd,
        'Adaptywny Hough': adaptive_hough,
        'Zoptymalizowane': optimized_hough,
        'Canny + LSD': canny_plus_lsd
    }
    
    results = {}
    performance = {}
    
    print("Testowanie metod detekcji linii...")
    print("-" * 50)
    
    for method_name, method_func in methods.items():
        start_time = time.time()
        lines = method_func(img)
        end_time = time.time()
        
        results[method_name] = lines
        performance[method_name] = {
            'lines_count': len(lines),
            'time_seconds': end_time - start_time,
            'lines_per_second': len(lines) / (end_time - start_time) if end_time > start_time else 0
        }
        
        print(f"{method_name:20} | Linii: {len(lines):4} | Czas: {end_time - start_time:.3f}s")
    
    return results, performance

def save_benchmark_report(performance, img_shape, save_path):
    """Zapisz raport z benchmarku"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'image_shape': img_shape,
        'methods_performance': performance,
        'best_method': max(performance.keys(), key=lambda k: performance[k]['lines_count']),
        'fastest_method': max(performance.keys(), key=lambda k: performance[k]['lines_per_second'])
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nRaport zapisany: {save_path}")
    print(f"Najlepsza metoda (najwiecej linii): {report['best_method']}")
    print(f"Najszybsza metoda: {report['fastest_method']}")

def main():
    """Główna funkcja testująca"""
    # Ścieżki
    project_dir = Path(r"C:\python\Intelligent_Sketch2CAD")
    intermediate_dir = project_dir / "intermediate_data"
    
    # Użyj konkretnego pliku adaptive_cleaned_20260509_115516.jpg
    target_image = intermediate_dir / "adaptive_cleaned_20260509_115516.jpg"
    
    if not target_image.exists():
        print(f"Nie znaleziono obrazu: {target_image}")
        print("Dostępne obrazy w intermediate_data:")
        pattern = str(intermediate_dir / "*.jpg")
        files = glob.glob(pattern)
        for img in files:
            print(f"  - {Path(img).name}")
        return
    
    original_image = str(target_image)
    print(f"Testowanie na obrazie: {Path(original_image).name}")
    
    # Wczytaj obraz
    img = load_test_image(original_image)
    
    # Benchmark wszystkich metod
    results, performance = benchmark_methods(img)
    
    # Wizualizacja wyników
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = intermediate_dir / f"line_comparison_{timestamp}.jpg"
    visualize_results(img, results, save_path=str(vis_path))
    
    # Zapisz raport
    report_path = intermediate_dir / f"benchmark_report_{timestamp}.json"
    save_benchmark_report(performance, img.shape, report_path)
    
    # Podsumowanie w konsoli
    print("\n" + "="*60)
    print("PODSUMOWANIE BENCHMARKU")
    print("="*60)
    print(f"Obraz: {Path(original_image).name} ({img.shape})")
    print("-" * 60)
    
    for method, stats in performance.items():
        print(f"{method:20} | Linii: {stats['lines_count']:4} | "
              f"Czas: {stats['time_seconds']:.3f}s | "
              f"Linii/s: {stats['lines_per_second']:.1f}")
    
    print("-" * 60)
    best_method = max(performance.keys(), key=lambda k: performance[k]['lines_count'])
    fastest_method = max(performance.keys(), key=lambda k: performance[k]['lines_per_second'])
    
    print(f"\n*** NAJLEPSZA METODA (najwiecej linii): {best_method}")
    print(f"*** NAJSZYBSZA METODA: {fastest_method}")

if __name__ == "__main__":
    main()
