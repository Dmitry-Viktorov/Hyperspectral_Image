import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

try:
    import xarray as xr
except ImportError:
    print('Ошибка: xarray не установлен. Установите через `pip install -r requirements.txt`.')
    sys.exit(1)

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('Ошибка: GDAL не установлен. Установите локальное колесо или через conda.')
    sys.exit(1)

DATA_DIR = 'dataset'
DATA_FILE = os.path.join(DATA_DIR, 'AV320241005t175313_005_L1B_RDN_f6524bfe_RDN.nc')
GROUP_NAME = 'radiance'
OUTPUT_TIFF = os.path.join(DATA_DIR, os.path.basename(DATA_FILE).replace('.nc', '.tif'))
REPORTS_DIR = 'reports'
METADATA_REPORT = os.path.join(REPORTS_DIR, 'metadata_report.txt')
SUMMARY_REPORT = os.path.join(REPORTS_DIR, 'metadata_summary.txt')


def print_progress(prefix, current, total):
    percent = int((current / total) * 100)
    bar_len = 30
    filled = int(bar_len * percent / 100)
    bar = '[' + '=' * filled + ' ' * (bar_len - filled) + ']' + f' {percent:3d}%'
    print(f'{prefix} {bar}', end='\r', flush=True)
    if current == total:
        print()


def normalize_label(text):
    if text is None:
        return ''
    return re.sub(r'[^0-9a-z]+', ' ', str(text).lower()).strip()


def flatten_xml(element, parent=''):
    path = f'{parent}/{element.tag}' if parent else element.tag
    data = {}
    for name, value in element.attrib.items():
        data[f'{path}/@{name}'] = value
    text = (element.text or '').strip()
    if text and len(element) == 0:
        data[path] = text
    for child in element:
        data.update(flatten_xml(child, path))
    return data


def parse_rsp_file(file_path):
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as exc:
        print(f'Предупреждение: не удалось разобрать {file_path}: {exc}')
        return {}
    root = tree.getroot()
    return flatten_xml(root)


def get_dataset_metadata(ds):
    keys = set()
    values = set()

    def add_attrs(obj):
        # Нормализуем ключи/значения, чтобы корректно сравнивать разные источники метаданных.
        for name, value in obj.attrs.items():
            keys.add(normalize_label(name))
            if isinstance(value, (list, tuple, np.ndarray)):
                values.update(normalize_label(v) for v in np.asarray(value).flatten() if v is not None)
            else:
                values.add(normalize_label(value))

    add_attrs(ds)
    for coord_name, coord_obj in ds.coords.items():
        keys.add(normalize_label(coord_name))
        add_attrs(coord_obj)
    for var_name, var_obj in ds.data_vars.items():
        keys.add(normalize_label(var_name))
        add_attrs(var_obj)
    if 'wavelength' in ds:
        values.update(normalize_label(v) for v in np.asarray(ds['wavelength'].values).flatten())
    return keys, values


def get_rsp_metadata(rsp_data):
    keys = set()
    values = set()
    for path, value in rsp_data.items():
        normalized_path = normalize_label(path)
        keys.add(normalized_path)
        segment = normalize_label(path.split('/')[-1])
        keys.add(segment)
        if isinstance(value, str):
            values.add(normalize_label(value))
            for item in value.split(','):
                item = item.strip()
                if item:
                    values.add(normalize_label(item))
        else:
            values.add(normalize_label(value))
    return keys, values


def compare_metadata(ds, rsp_data):
    ds_keys, ds_values = get_dataset_metadata(ds)
    rsp_keys, rsp_values = get_rsp_metadata(rsp_data)

    exact_keys = sorted(ds_keys & rsp_keys)
    exact_values = sorted(ds_values & rsp_values)

    approx_pairs = set()
    for ds_key in ds_keys:
        for rsp_key in rsp_keys:
            if ds_key and rsp_key and (ds_key in rsp_key or rsp_key in ds_key):
                if ds_key != rsp_key:
                    approx_pairs.add((ds_key, rsp_key))
    approximate_keys = sorted(approx_pairs)
    return exact_keys, exact_values, approximate_keys


def get_dataset_georef(ds):
    georef = {}
    if 'geotransform' in ds.attrs:
        georef['geotransform'] = ds.attrs.get('geotransform')
    if 'crs' in ds.coords:
        georef['crs'] = ds.coords['crs'].attrs.get('spatial_ref', None) or ds.coords['crs'].attrs.get('crs_wkt', None)
    if georef:
        return georef

    x_names = ['x', 'longitude', 'lon']
    y_names = ['y', 'latitude', 'lat']
    for x_name in x_names:
        for y_name in y_names:
            if x_name in ds.coords and y_name in ds.coords:
                x = ds.coords[x_name].values
                y = ds.coords[y_name].values
                if x.ndim == 1 and y.ndim == 1 and x.size > 1 and y.size > 1:
                    dx = float(np.mean(np.diff(x)))
                    dy = float(np.mean(np.diff(y)))
                    # Геотрансформ строим только для почти равномерной сетки координат.
                    if abs(np.std(np.diff(x))) < abs(dx) * 1e-3 and abs(np.std(np.diff(y))) < abs(dy) * 1e-3:
                        ulx = float(x[0] - dx / 2)
                        uly = float(y[0] - dy / 2) if dy < 0 else float(y[0] + dy / 2)
                        georef['geotransform'] = (ulx, dx, 0.0, uly, 0.0, dy)
                        georef['crs'] = ds.coords[y_name].attrs.get('spatial_ref', None) or ds.coords[y_name].attrs.get('crs_wkt', None)
                        return georef
    return georef


def read_tiff_metadata(file_path):
    tiff_data = {}
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        return {'error': f'Failed to open TIFF {file_path}'}

    tiff_data['driver'] = dataset.GetDriver().LongName
    tiff_data['size'] = (dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount)
    tiff_data['projection'] = dataset.GetProjection() or 'None'
    try:
        gt = dataset.GetGeoTransform(can_return_null=True)
    except TypeError:
        gt = dataset.GetGeoTransform()
    tiff_data['geotransform'] = gt
    tiff_data['metadata'] = dataset.GetMetadata() or {}

    bands = []
    for band_index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(band_index)
        bands.append({
            'description': band.GetDescription() or '',
            'metadata': band.GetMetadata() or {},
            'dtype': gdal.GetDataTypeName(band.DataType),
            'nodata': band.GetNoDataValue(),
        })
    tiff_data['bands'] = bands
    dataset = None
    return tiff_data


def get_tiff_metadata_sets(tiff_info):
    keys = set()
    values = set()
    if not tiff_info or 'error' in tiff_info:
        return keys, values

    keys.update(
        normalize_label(k) for k in ['driver', 'size', 'projection', 'geotransform', 'bands', 'metadata']
    )
    values.update(
        normalize_label(v) for v in [
            tiff_info.get('driver'),
            tiff_info.get('projection'),
            tiff_info.get('geotransform'),
            tiff_info.get('size'),
        ]
    )

    for name, value in tiff_info.get('metadata', {}).items():
        keys.add(normalize_label(name))
        values.add(normalize_label(value))

    for band in tiff_info.get('bands', []):
        keys.add(normalize_label('band_description'))
        keys.add(normalize_label('band_dtype'))
        values.add(normalize_label(band.get('description')))
        values.add(normalize_label(band.get('dtype')))
        for name, value in band.get('metadata', {}).items():
            keys.add(normalize_label(name))
            values.add(normalize_label(value))

    return keys, values


def get_rsp_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith('.rsp'))


def collect_dataset_summary(ds):
    summary = {
        'bands': int(ds['radiance'].shape[0]),
        'shape': tuple(int(v) for v in ds['radiance'].shape),
        'radiance_units': ds['radiance'].attrs.get('units', 'unknown'),
        'fwhm_present': 'fwhm' in ds.data_vars,
        'crs': get_dataset_georef(ds).get('crs', None),
    }
    if 'wavelength' in ds:
        wavelengths = np.asarray(ds['wavelength'].values, dtype=float)
        summary.update({
            'wavelength_count': int(wavelengths.size),
            'wavelength_min': float(np.min(wavelengths)),
            'wavelength_max': float(np.max(wavelengths)),
        })
    else:
        summary['wavelength_count'] = None
    return summary


def write_summary_report(ds, rsp_results, output_path, tiff_info=None, rsp_union_keys=None, rsp_union_values=None):
    lines = []
    summary = collect_dataset_summary(ds)
    nc_keys, nc_values = get_dataset_metadata(ds)
    tiff_keys, tiff_values = get_tiff_metadata_sets(tiff_info)
    rsp_union_keys = rsp_union_keys or set()
    rsp_union_values = rsp_union_values or set()

    lines.append('Краткое резюме метаданных')
    lines.append('=' * 28)
    lines.append(f"Гиперспектральный куб: {summary['shape']} (bands={summary['bands']})")
    if summary['wavelength_count'] is not None:
        lines.append(f"Диапазон длин волн: {summary['wavelength_min']:.3f}–{summary['wavelength_max']:.3f} nm")
    lines.append(f"Единицы радиации: {summary['radiance_units']}")
    lines.append(f"FWHM-присутствует: {'да' if summary['fwhm_present'] else 'нет'}")
    lines.append(f"CRS: {summary['crs'] or 'не найден'}")
    lines.append('')
    lines.append(f"Найдено .rsp файлов: {len(rsp_results)}")
    if rsp_results:
        for rsp in rsp_results:
            lines.append(f"  - {rsp['file']}: {rsp['entries']} записей, точных ключей={len(rsp['exact_keys'])}, точных значений={len(rsp['exact_values'])}")
    lines.append('')
    if rsp_results:
        all_exact_keys = sorted({k for rsp in rsp_results for k in rsp['exact_keys']})
        all_exact_values = sorted({v for rsp in rsp_results for v in rsp['exact_values']})
        lines.append(f"Всего общих ключей: {len(all_exact_keys)}")
        lines.append(f"Всего общих значений: {len(all_exact_values)}")
        if all_exact_keys:
            lines.append('Примеры общих ключей:')
            for key in all_exact_keys[:10]:
                lines.append(f'  {key}')
        if all_exact_values:
            lines.append('Примеры общих значений:')
            for value in all_exact_values[:10]:
                lines.append(f'  {value}')

    lines.append('')
    lines.append('Пересечения между источниками (.nc/.tif/.rsp):')
    if tiff_keys or tiff_values:
        nc_tiff_keys = sorted(nc_keys & tiff_keys)
        nc_tiff_values = sorted(nc_values & tiff_values)
        tiff_rsp_keys = sorted(tiff_keys & rsp_union_keys)
        tiff_rsp_values = sorted(tiff_values & rsp_union_values)
        triple_keys = sorted(nc_keys & tiff_keys & rsp_union_keys)
        triple_values = sorted(nc_values & tiff_values & rsp_union_values)

        lines.append(f'  NC <-> TIFF: ключи={len(nc_tiff_keys)}, значения={len(nc_tiff_values)}')
        lines.append(f'  TIFF <-> RSP: ключи={len(tiff_rsp_keys)}, значения={len(tiff_rsp_values)}')
        lines.append(f'  NC <-> TIFF <-> RSP: ключи={len(triple_keys)}, значения={len(triple_values)}')
    else:
        lines.append('  TIFF не найден или не прочитан, пересечения с TIFF не вычислялись.')

    lines.append('')
    with open(output_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))
    print(f'Краткий отчёт сохранён в: {output_path}')


def write_metadata_report(ds, output_path, tiff_path=OUTPUT_TIFF):
    lines = []
    nc_keys, nc_values = get_dataset_metadata(ds)

    lines.append('Отчёт по метаданным')
    lines.append('=' * 24)
    lines.append('')
    lines.append('Файл: ' + DATA_FILE)
    lines.append('Группа: ' + GROUP_NAME)
    lines.append('')
    lines.append('Размерности:')
    for name, size in ds.sizes.items():
        lines.append(f'  {name}: {size}')
    lines.append('')

    lines.append('Координаты:')
    for coord in ds.coords:
        coord_obj = ds.coords[coord]
        lines.append(f'  {coord}: dtype={coord_obj.dtype}, shape={coord_obj.shape}')
        for attr_name, attr_value in coord_obj.attrs.items():
            lines.append(f'    {attr_name}: {attr_value}')
    lines.append('')

    lines.append('Переменные данных:')
    for var in ds.data_vars:
        variable = ds[var]
        lines.append(f'  {var}: dtype={variable.dtype}, shape={variable.shape}')
        for attr_name, attr_value in variable.attrs.items():
            lines.append(f'    {attr_name}: {attr_value}')
    lines.append('')

    lines.append('Глобальные атрибуты:')
    if ds.attrs:
        for name, value in ds.attrs.items():
            lines.append(f'  {name}: {value}')
    else:
        lines.append('  (глобальные атрибуты отсутствуют)')
    lines.append('')

    georef = get_dataset_georef(ds)
    lines.append('Геопривязка и CRS:')
    if georef:
        for key, value in georef.items():
            lines.append(f'  {key}: {value}')
    else:
        lines.append('  (геопривязка явно не указана в датасете)')
    lines.append('')

    rsp_files = get_rsp_files()
    lines.append('Найденные .rsp файлы: ' + (', '.join(rsp_files) if rsp_files else 'Нет'))
    lines.append('')

    all_exact_keys = set()
    all_exact_values = set()
    all_approximate = set()
    rsp_union_keys = set()
    rsp_union_values = set()

    for idx, rsp_file in enumerate(rsp_files, start=1):
        print_progress('Processing RSP files:', idx, len(rsp_files))
        rsp_path = os.path.abspath(rsp_file)
        rsp_data = parse_rsp_file(rsp_path)
        rsp_keys, rsp_values = get_rsp_metadata(rsp_data)
        rsp_union_keys |= rsp_keys
        rsp_union_values |= rsp_values
        exact_keys, exact_values, approximate_keys = compare_metadata(ds, rsp_data)
        lines.append(f'RSP файл: {rsp_file}')
        lines.append(f'  Количество записей: {len(rsp_data)}')
        keys = [k for k in sorted(rsp_data.keys()) if k.endswith('/@val')]
        for key in keys[:20]:
            lines.append(f'    {key}: {rsp_data[key]}')
        if len(keys) > 20:
            lines.append(f'    ... ({len(keys) - 20} more entries)')

        if exact_keys:
            lines.append('  Точные совпадения ключей:')
            for key in exact_keys:
                lines.append(f'    {key}')
            all_exact_keys |= set(exact_keys)
        else:
            lines.append('  Точные совпадения ключей: нет')

        if exact_values:
            lines.append('  Точные совпадения значений:')
            for value in exact_values[:20]:
                lines.append(f'    {value}')
            all_exact_values |= set(exact_values)
        else:
            lines.append('  Точные совпадения значений: нет')

        if approximate_keys:
            lines.append('  Вероятные схожие ключи:')
            for ds_key, rsp_key in approximate_keys[:20]:
                lines.append(f'    {ds_key} <-> {rsp_key}')
            all_approximate |= set(approximate_keys)
        else:
            lines.append('  Вероятные схожие ключи: нет')

        lines.append('')

    if not rsp_files:
        lines.append('Примечание: .rsp файлы не найдены. Сравнение возможно при наличии паспортов.')
    else:
        lines.append('Итог сравнения с .rsp файлами:')
        lines.append(f'  Всего точных ключей: {len(all_exact_keys)}')
        lines.append(f'  Всего точных значений: {len(all_exact_values)}')
        lines.append(f'  Всего приблизительных совпадений ключей: {len(all_approximate)}')
    lines.append('')

    if tiff_path is None:
        lines.append('Метаданные TIFF после экспорта: пропущены, так как экспорт TIFF был отключён.')
    else:
        tiff_info = read_tiff_metadata(tiff_path)
        tiff_keys, tiff_values = get_tiff_metadata_sets(tiff_info)
        lines.append('TIFF metadata after export:')
        if 'error' in tiff_info:
            lines.append(f"  {tiff_info['error']}")
        else:
            lines.append(f"  driver: {tiff_info['driver']}")
            lines.append(f"  size: {tiff_info['size']}")
            lines.append(f"  projection: {tiff_info['projection']}")
            lines.append(f"  geotransform: {tiff_info['geotransform']}")
            if tiff_info['metadata']:
                lines.append('  метаданные:')
                for name, value in tiff_info['metadata'].items():
                    lines.append(f'    {name}: {value}')
            else:
                lines.append('  метаданные: отсутствуют')
            lines.append('  полосы:')
            for idx, band in enumerate(tiff_info['bands'], start=1):
                lines.append(f"    полоса {idx}: dtype={band['dtype']}, description={band['description']}")
                if band['metadata']:
                    for name, value in band['metadata'].items():
                        lines.append(f'      {name}: {value}')

            lines.append('')
            lines.append('Пересечения между источниками (.nc/.tif/.rsp):')
            nc_tiff_keys = sorted(nc_keys & tiff_keys)
            nc_tiff_values = sorted(nc_values & tiff_values)
            tiff_rsp_keys = sorted(tiff_keys & rsp_union_keys)
            tiff_rsp_values = sorted(tiff_values & rsp_union_values)
            triple_keys = sorted(nc_keys & tiff_keys & rsp_union_keys)
            triple_values = sorted(nc_values & tiff_values & rsp_union_values)

            lines.append(f'  NC <-> TIFF: ключи={len(nc_tiff_keys)}, значения={len(nc_tiff_values)}')
            lines.append(f'  TIFF <-> RSP: ключи={len(tiff_rsp_keys)}, значения={len(tiff_rsp_values)}')
            lines.append(f'  NC <-> TIFF <-> RSP: ключи={len(triple_keys)}, значения={len(triple_values)}')

            if nc_tiff_keys:
                lines.append('  Примеры NC <-> TIFF ключей:')
                for key in nc_tiff_keys[:10]:
                    lines.append(f'    {key}')
            if tiff_rsp_keys:
                lines.append('  Примеры TIFF <-> RSP ключей:')
                for key in tiff_rsp_keys[:10]:
                    lines.append(f'    {key}')
            if triple_keys:
                lines.append('  Примеры общих ключей NC/TIFF/RSP:')
                for key in triple_keys[:10]:
                    lines.append(f'    {key}')

            if nc_tiff_values:
                lines.append('  Примеры NC <-> TIFF значений:')
                for value in nc_tiff_values[:10]:
                    lines.append(f'    {value}')
            if tiff_rsp_values:
                lines.append('  Примеры TIFF <-> RSP значений:')
                for value in tiff_rsp_values[:10]:
                    lines.append(f'    {value}')
            if triple_values:
                lines.append('  Примеры общих значений NC/TIFF/RSP:')
                for value in triple_values[:10]:
                    lines.append(f'    {value}')
    lines.append('')

    with open(output_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))

    print(f'Отчёт сохранён в: {output_path}')


def export_to_tiff(ds, output_path):
    radiance = ds['radiance']
    wavelength = ds.get('wavelength', None)
    nbands, nlines, nsamples = radiance.shape

    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES', 'PREDICTOR=2']
    if os.path.exists(output_path):
        os.remove(output_path)
    dataset = driver.Create(output_path, nsamples, nlines, nbands, gdal.GDT_Float32, options)
    if dataset is None:
        raise RuntimeError('Failed to create output TIFF dataset')

    if ds.attrs:
        metadata = {str(k): str(v) for k, v in ds.attrs.items()}
        dataset.SetMetadata(metadata)

    georef = get_dataset_georef(ds)
    if 'geotransform' in georef:
        try:
            dataset.SetGeoTransform(tuple(georef['geotransform']))
        except Exception:
            pass
    if 'crs' in georef and georef['crs']:
        dataset.SetProjection(str(georef['crs']))

    if wavelength is not None:
        dataset.SetMetadataItem('WAVELENGTH_COUNT', str(len(wavelength)))

    print(f'Начало экспорта TIFF ({nbands} полос):')
    for band_index in range(nbands):
        band = dataset.GetRasterBand(band_index + 1)
        band_array = radiance[band_index, :, :].values.astype(np.float32)
        band.WriteArray(band_array)
        band.FlushCache()

        if wavelength is not None:
            wavelength_value = float(wavelength.values[band_index])
            band.SetDescription(f'Wavelength {wavelength_value:.3f}')
            band.SetMetadataItem('WAVELENGTH', str(wavelength_value))

        print_progress('Запись TIFF-полос:', band_index + 1, nbands)

    dataset.FlushCache()
    print(f'Экспорт TIFF завершён: {output_path}')


def plot_dataset(ds, plot_dir, show_plots=True, save_plots=True):
    radiance = ds['radiance']
    wavelength = ds.get('wavelength', None)
    os.makedirs(plot_dir, exist_ok=True)

    band_index = 50
    if band_index >= radiance.shape[0]:
        band_index = int(radiance.shape[0] // 2)
        print(f'Band index 50 out of range, using {band_index} instead.')

    image = radiance[band_index, :, :].values
    plt.figure(figsize=(8, 6))
    title = f'Band {band_index}'
    if wavelength is not None:
        title += f' ({wavelength.values[band_index]:.2f} nm)'
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.colorbar(label='Radiance')
    plt.tight_layout()
    band_path = os.path.join(plot_dir, f'Band{band_index + 1}.png')
    if save_plots:
        plt.savefig(band_path, dpi=150)
        print(f'График полосы сохранён: {band_path}')
    if show_plots:
        plt.show()
    plt.close()

    y = min(100, radiance.shape[1] - 1)
    x = min(200, radiance.shape[2] - 1)
    spectrum = radiance[:, y, x].values

    plt.figure(figsize=(8, 4))
    if wavelength is not None:
        plt.plot(wavelength.values, spectrum)
        plt.xlabel('Wavelength, nm')
    else:
        plt.plot(spectrum)
        plt.xlabel('Band index')
    plt.ylabel('Radiance')
    plt.title(f'Spectrum at pixel (x={x}, y={y})')
    plt.grid(True)
    plt.tight_layout()
    spectrum_path = os.path.join(plot_dir, f'SpectralPixel_x{x}_y{y}.png')
    if save_plots:
        plt.savefig(spectrum_path, dpi=150)
        print(f'График спектра сохранён: {spectrum_path}')
    if show_plots:
        plt.show()
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Распаковка гиперспектрального NetCDF в TIFF и формирование отчёта метаданных')
    parser.add_argument('--no-plot', action='store_true', help='Не строить и не сохранять графики')
    parser.add_argument('--no-tiff', action='store_true', help='Не экспортировать TIFF')
    parser.add_argument('--no-report', action='store_true', help='Не генерировать детальный отчёт метаданных')
    parser.add_argument('--no-summary', action='store_true', help='Не генерировать краткий отчёт')
    parser.add_argument('--output-tiff', default=OUTPUT_TIFF, help='Имя выходного TIFF-файла')
    parser.add_argument('--report-file', default=METADATA_REPORT, help='Имя файла детального отчёта метаданных')
    parser.add_argument('--summary-file', default=SUMMARY_REPORT, help='Имя файла краткого отчёта метаданных')
    parser.add_argument('--save-plots', dest='save_plots', action='store_true', help='Сохранить графики в reports/')
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false', help='Не сохранять графики')
    parser.add_argument('--show-plots', action='store_true', help='Показать графики интерактивно')
    parser.set_defaults(save_plots=True)
    args = parser.parse_args(argv)

    if not os.path.exists(DATA_FILE):
        print(f'Error: file not found: {DATA_FILE}')
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f'Открытие датасета: {DATA_FILE} (группа={GROUP_NAME})')
    ds = xr.open_dataset(DATA_FILE, group=GROUP_NAME)
    print('Датасет успешно загружен')

    print('Размерности:')
    for name, size in ds.sizes.items():
        print(f'  {name}: {size}')
    print('\nПеременные:')
    for var in ds.data_vars:
        print(f'  {var}')

    if 'radiance' not in ds:
        print('Ошибка: переменная `radiance` не найдена в Датасете')
        return

    if not args.no_tiff:
        export_to_tiff(ds, args.output_tiff)

    # Даже если экспорт отключен, берём существующий TIFF для сравнения метаданных.
    tiff_path_for_analysis = args.output_tiff if os.path.exists(args.output_tiff) else None
    tiff_info_for_summary = read_tiff_metadata(tiff_path_for_analysis) if tiff_path_for_analysis else None

    rsp_files = get_rsp_files()
    rsp_results = []
    rsp_union_keys = set()
    rsp_union_values = set()
    for rsp_file in rsp_files:
        rsp_data = parse_rsp_file(rsp_file)
        rsp_keys, rsp_values = get_rsp_metadata(rsp_data)
        rsp_union_keys |= rsp_keys
        rsp_union_values |= rsp_values
        exact_keys, exact_values, approximate_keys = compare_metadata(ds, rsp_data)
        rsp_results.append({
            'file': rsp_file,
            'entries': len(rsp_data),
            'exact_keys': exact_keys,
            'exact_values': exact_values,
            'approximate_keys': approximate_keys,
        })

    if not args.no_report:
        print('Generating metadata report...')
        write_metadata_report(ds, args.report_file, tiff_path_for_analysis)

    if not args.no_summary:
        write_summary_report(
            ds,
            rsp_results,
            args.summary_file,
            tiff_info=tiff_info_for_summary,
            rsp_union_keys=rsp_union_keys,
            rsp_union_values=rsp_union_values,
        )

    if not args.no_plot:
        plot_dataset(ds, REPORTS_DIR, show_plots=args.show_plots, save_plots=args.save_plots)


if __name__ == '__main__':
    main()
