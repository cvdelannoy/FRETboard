function str2bytes (str) {
    var bytes = new Uint8Array(str.length);
    for (var i=0; i<str.length; i++) {
        bytes[i] = str.charCodeAt(i);
    }
    return bytes;
}

const filename = 'dat_files.zip';
var b64data = window.atob(file_source.data.datzip[0]);
const blob = new Blob([str2bytes(b64data)], { type: 'application/octet-stream' });
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