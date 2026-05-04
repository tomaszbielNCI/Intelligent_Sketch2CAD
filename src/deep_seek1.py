import cadquery as cq
from pathlib import Path


def generate_glass_wall(width=1200, height=2000, glass_thick=12,
                        door_width=800, door_side="right"):
    """
    Generuje szklaną ściankę z drzwiami

    Args:
        width: szerokość całkowita [mm]
        height: wysokość [mm]
        glass_thick: grubość szkła [mm]
        door_width: szerokość drzwi [mm]
        door_side: strona drzwi ("left"/"right")

    Returns:
        cq.Assembly
    """
    # Główna ściana (czysty obszar)
    main_wall = cq.Workplane("XY").box(width, height, glass_thick)

    if door_side == "left":
        door_x_start = 0
    else:
        door_x_start = width - door_width

    # Wycięcie na drzwi
    door_cutout = cq.Workplane("XY").box(door_width, height, glass_thick + 1) \
        .translate((door_x_start + door_width / 2, 0, 0))

    final_wall = main_wall.cut(door_cutout)

    # Opcjonalnie: dodaj ramę (dla realizmu)
    frame = cq.Workplane("XY").box(width, 40, 20) \
        .translate((0, -height / 2 + 20, glass_thick / 2))

    assembly = cq.Assembly()
    assembly.add(final_wall, name="glass_wall")
    assembly.add(frame, name="frame")

    return assembly


def export_to_step(assembly, filename="output/glass_wall.step"):
    """Eksport do formatu STEP"""
    Path("output").mkdir(exist_ok=True)
    cq.exporters.export(assembly, filename)
    return filename