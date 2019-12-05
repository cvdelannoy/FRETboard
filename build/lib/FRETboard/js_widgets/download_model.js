const filename = 'FRETboard_model.txt';
const columns = Object.keys(file_source.data);
const blob = new Blob([file_source.data[columns[0]][0].toString()], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.target = '_blank';
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}