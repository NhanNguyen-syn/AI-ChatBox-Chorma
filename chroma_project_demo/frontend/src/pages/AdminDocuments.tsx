import React, { useEffect, useMemo, useRef, useState } from 'react'
import { UploadCloud, Trash2, RefreshCcw, FileSpreadsheet, X, AlertTriangle } from 'lucide-react'
import { api } from '../services/api'
import toast from 'react-hot-toast'

interface DocItem {
  id: string
  filename: string
  file_type: string
  uploaded_by: string
  uploaded_at: string
  file_size: number
}

interface ExcelItem {
  id: string
  filename: string
  file_size: number
  uploaded_at: string
  sheet_names: string[]
  rows_processed: number
  collection_name: string
}

const formatBytes = (bytes: number) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const AdminDocuments: React.FC = () => {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [excel, setExcel] = useState<ExcelItem[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadingExcel, setUploadingExcel] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const excelInputRef = useRef<HTMLInputElement>(null)
  const [docPage, setDocPage] = useState(1)
  const [docTotal, setDocTotal] = useState(0)
  const DOCS_PER_PAGE = 10

  const [isConfirmOpen, setConfirmOpen] = useState(false)
  const [docsToDelete, setDocsToDelete] = useState<string[]>([])
  const [deleting, setDeleting] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, { progress: number; status: 'uploading' | 'processing' | 'success' | 'error'; error?: string }>>({})
  const [searchQuery, setSearchQuery] = useState('')

  const filteredDocs = useMemo(() => {
    if (!searchQuery.trim()) return docs
    const q = searchQuery.toLowerCase()
    return docs.filter(d =>
      d.filename.toLowerCase().includes(q) ||
      d.uploaded_by.toLowerCase().includes(q)
    )
  }, [docs, searchQuery])

  const loadDocs = async () => {
    setLoading(true)
    try {
      const [docsRes, excelRes] = await Promise.all([
        api.get(`/files/documents?page=${docPage}&limit=${DOCS_PER_PAGE}`),
        api.get('/admin/excel-files') // Keep excel loading as is for now
      ])
      setDocs(docsRes.data.documents ?? [])
      setDocTotal(docsRes.data.total ?? 0)
      setExcel(Array.isArray(excelRes.data) ? excelRes.data : (excelRes.data ?? []))
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Không thể tải danh sách dữ liệu')
      setDocs([])
      setExcel([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocs()
  }, [docPage])

  const uploadViaChunks = async (file: File) => {
    const CHUNK_SIZE = 5 * 1024 * 1024 // 5MB
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE)
    const uploadId = `${Date.now()}_${file.name}`

    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE
      const end = Math.min(start + CHUNK_SIZE, file.size)
      const blob = file.slice(start, end)
      const form = new FormData()
      form.append('file', blob, file.name)
      form.append('upload_id', uploadId)
      form.append('index', String(i))
      form.append('total', String(totalChunks))
      form.append('filename', file.name)
      await api.post('/files/upload-chunk', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      const progress = Math.round(((i + 1) / totalChunks) * 100)
      setUploadProgress(prev => ({ ...prev, [file.name]: { ...prev[file.name], progress, status: 'uploading' } }))
    }

    setUploadProgress(prev => ({ ...prev, [file.name]: { ...prev[file.name], status: 'processing', progress: 100 } }))
    const fin = new FormData()
    fin.append('upload_id', uploadId)
    fin.append('filename', file.name)
    fin.append('total', String(totalChunks))
    await api.post('/files/upload-chunk/finish', fin, { headers: { 'Content-Type': 'multipart/form-data' } })
  }

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    setUploading(true)
    const files = Array.from(fileList)

    const initialProgress = files.reduce((acc, file) => {
      acc[file.name] = { progress: 0, status: 'uploading', error: undefined }
      return acc
    }, {} as typeof uploadProgress)
    setUploadProgress(prev => ({ ...prev, ...initialProgress }))

    const uploadPromises = files.map(async (file) => {
      try {
        if (file.size > 15 * 1024 * 1024) {
          await uploadViaChunks(file)
        } else {
          const form = new FormData()
          form.append('file', file)
          await api.post('/files/upload', form, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (e) => {
              const progress = Math.round((e.loaded * 100) / (e.total || file.size))
              setUploadProgress(prev => ({ ...prev, [file.name]: { ...prev[file.name], progress, status: 'uploading' } }))
            }
          })
          setUploadProgress(prev => ({ ...prev, [file.name]: { ...prev[file.name], status: 'processing', progress: 100 } }))
        }
        setUploadProgress(prev => ({ ...prev, [file.name]: { ...prev[file.name], status: 'success' } }))
      } catch (e: any) {
        const errorMsg = e.response?.data?.detail || 'Upload thất bại'
        toast.error(`${file.name}: ${errorMsg}`)
        setUploadProgress(prev => ({ ...prev, [file.name]: { ...prev[file.name], status: 'error', error: errorMsg } }))
      }
    })

    await Promise.all(uploadPromises)
    await loadDocs()
    setUploading(false)
    toast.success('Hoàn tất quá trình upload.')

    setTimeout(() => {
      setUploadProgress(prev => {
        const next = { ...prev }
        Object.keys(next).forEach(key => {
          if (next[key].status === 'success' || next[key].status === 'error') delete next[key]
        })
        return next
      })
    }, 5000)
  }

  const handleExcelUpload = async (file: File | null) => {
    if (!file) return
    setUploadingExcel(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await api.post('/admin/upload-excel', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      toast.success(`Đã xử lý Excel: ${res.data?.filename}`)
      await loadDocs()
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Upload Excel thất bại')
    } finally {
      setUploadingExcel(false)
    }
  }

  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault()
    e.stopPropagation()
    handleFiles(e.dataTransfer.files)
  }

  const openConfirmDelete = (ids: string[]) => {
    setDocsToDelete(ids)
    setConfirmOpen(true)
  }

  const handleConfirmDelete = async () => {
    setDeleting(true)
    try {
      // Assuming a batch delete endpoint, otherwise loop
      await api.post('/files/documents/batch-delete', { ids: docsToDelete })
      toast.success(`Đã xóa ${docsToDelete.length} tài liệu`)
      setDocs(docs => docs.filter(d => !docsToDelete.includes(d.id)))
      setSelectedDocs([])
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Lỗi khi xóa tài liệu')
    } finally {
      setDeleting(false)
      setConfirmOpen(false)
      setDocsToDelete([])
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <>
      <div className="space-y-6">
        {/* Unstructured documents upload */}
        <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Quản lý dữ liệu</h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Upload tài liệu (PDF/DOC/DOCX/TXT/IMG) để lập chỉ mục và hỏi đáp</p>
            </div>
            <button
              onClick={loadDocs}
              className="btn-secondary inline-flex items-center gap-2"
            >
              <RefreshCcw className="h-4 w-4" /> Làm mới
            </button>
          </div>

          {/* Dropzone */}
          <div
            className="mt-6 rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 p-8 text-center hover:border-primary-400 dark:bg-[#0f0f0f] dark:border-gray-700"
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation() }}
            onDrop={onDrop}
          >
            <div className="flex flex-col items-center gap-3">
              <UploadCloud className="h-10 w-10 text-primary-600" />
              <p className="text-gray-700 dark:text-gray-300">Kéo thả tệp vào đây hoặc</p>
              <div>
                <button
                  onClick={() => inputRef.current?.click()}
                  className="btn-primary"
                  disabled={uploading}
                >{uploading ? 'Đang tải...' : 'Chọn tệp'}</button>
                <input
                  ref={inputRef}
                  type="file"
                  className="hidden"
                  multiple
                  accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
                  onChange={(e) => handleFiles(e.target.files)}
                />
              </div>
              <p className="text-xs text-gray-500">Hỗ trợ: PDF, DOC, DOCX, TXT, PNG, JPG</p>
            </div>
          </div>

          {/* Upload Progress */}
          {Object.keys(uploadProgress).length > 0 && (
            <div className="mt-4 space-y-3 pt-4 border-t dark:border-gray-700">
              {Object.entries(uploadProgress).map(([filename, { progress, status, error }]) => (
                <div key={filename}>
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="font-medium text-gray-800 dark:text-gray-200 truncate pr-4" title={filename}>{filename}</span>
                    <span className={`font-semibold ${status === 'success' ? 'text-green-600' :
                      status === 'error' ? 'text-red-600' :
                        'text-gray-500'
                      }`}>
                      {status === 'uploading' && `${progress}%`}
                      {status === 'processing' && 'Đang xử lý...'}
                      {status === 'success' && 'Thành công'}
                      {status === 'error' && 'Lỗi'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${status === 'success' ? 'bg-green-500' :
                        status === 'error' ? 'bg-red-500' :
                          'bg-primary-600'
                        }`}
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  {status === 'error' && <p className="text-xs text-red-600 mt-1">{error}</p>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Excel upload */}
        <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="h-6 w-6 text-green-600" />
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Upload Excel/CSV</h2>
                <p className="text-gray-600 dark:text-gray-400 mt-1">Hệ thống sẽ làm sạch, chuẩn hóa và lập chỉ mục để có thể hỏi đáp bằng ngôn ngữ tự nhiên</p>
              </div>
            </div>
            <div>
              <button onClick={() => excelInputRef.current?.click()} className="btn-primary" disabled={uploadingExcel}>
                {uploadingExcel ? 'Đang xử lý...' : 'Chọn Excel/CSV'}
              </button>
              <input
                ref={excelInputRef}
                type="file"
                className="hidden"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => handleExcelUpload(e.target.files?.[0] || null)}
              />
            </div>
          </div>
        </div>

        {/* List unstructured docs */}
        <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold dark:text-gray-100">Tài liệu đã tải lên</h2>
            {selectedDocs.length > 0 ? (
              <button onClick={() => openConfirmDelete(selectedDocs)} className="btn-danger inline-flex items-center gap-2">
                <Trash2 className="h-4 w-4" /> Xóa ({selectedDocs.length}) mục đã chọn
              </button>
            ) : (loading && <span className="text-sm text-gray-500">Đang tải...</span>)}
          </div>

          <div className="mb-4">
            <input
              type="text"
              placeholder="Tìm kiếm theo tên tệp hoặc người upload..."
              className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 dark:bg-zinc-800 dark:border-zinc-700"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {filteredDocs.length === 0 && !loading ? (
            <div className="text-gray-500 dark:text-gray-400">
              {searchQuery ? 'Không tìm thấy tài liệu phù hợp.' : 'Chưa có tài liệu nào.'}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-600 dark:text-gray-400 border-b dark:border-gray-700">
                    <th className="py-2 px-2 w-12 text-center">
                      <input type="checkbox"
                        className="rounded border-gray-300 text-primary-600 shadow-sm focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50"
                        checked={filteredDocs.length > 0 && selectedDocs.length === filteredDocs.length}
                        onChange={(e) => setSelectedDocs(e.target.checked ? filteredDocs.map((d: DocItem) => d.id) : [])}
                      />
                    </th>
                    <th className="py-2 pr-4">Tên tệp</th>
                    <th className="py-2 pr-4">Loại</th>
                    <th className="py-2 pr-4">Dung lượng</th>
                    <th className="py-2 pr-4">Người upload</th>
                    <th className="py-2 pr-4">Thời gian</th>
                    <th className="py-2 pr-4 text-right">Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDocs.map((d: DocItem) => (
                    <tr key={d.id} className="border-b last:border-0 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                      <td className="py-3 px-2 w-12 text-center">
                        <input type="checkbox"
                          className="rounded border-gray-300 text-primary-600 shadow-sm focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50"
                          checked={selectedDocs.includes(d.id)}
                          onChange={(e) => {
                            setSelectedDocs(e.target.checked ? [...selectedDocs, d.id] : selectedDocs.filter(id => id !== d.id))
                          }}
                        />
                      </td>
                      <td className="py-3 pr-4 font-medium text-gray-900 dark:text-gray-100">{d.filename}</td>
                      <td className="py-3 pr-4 uppercase dark:text-gray-300">{d.file_type}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{formatBytes(d.file_size)}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{d.uploaded_by}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{new Date(d.uploaded_at).toLocaleString('vi-VN')}</td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => openConfirmDelete([d.id])}
                            className="inline-flex items-center gap-1 text-red-600 hover:text-red-700 px-2 py-1 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20"
                          >
                            <Trash2 className="h-4 w-4" /> Xóa
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {docTotal > DOCS_PER_PAGE && (
            <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-800">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Hiển thị {(docPage - 1) * DOCS_PER_PAGE + 1} - {Math.min(docPage * DOCS_PER_PAGE, docTotal)} trên {docTotal}
              </span>
              <div className="space-x-2">
                <button
                  onClick={() => setDocPage(p => p - 1)}
                  disabled={docPage === 1}
                  className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Trang trước
                </button>
                <button
                  onClick={() => setDocPage(p => p + 1)}
                  disabled={docPage * DOCS_PER_PAGE >= docTotal}
                  className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Trang sau
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Excel datasets list */}
        <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold dark:text-gray-100">Bộ dữ liệu Excel</h2>
          </div>
          {excel.length === 0 ? (
            <div className="text-gray-500 dark:text-gray-400">Chưa có dữ liệu Excel nào.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-600 dark:text-gray-400 border-b dark:border-gray-700">
                    <th className="py-2 pr-4">Tên tệp</th>
                    <th className="py-2 pr-4">Sheets</th>
                    <th className="py-2 pr-4">Số dòng</th>
                    <th className="py-2 pr-4">Dung lượng</th>
                    <th className="py-2 pr-4">Thời gian</th>
                    <th className="py-2 pr-4">Collection</th>
                    <th className="py-2 pr-4 text-right">Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {excel.map((e) => (
                    <tr key={e.id} className="border-b last:border-0 dark:border-gray-700">
                      <td className="py-3 pr-4 font-medium text-gray-900 dark:text-gray-100">{e.filename}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{e.sheet_names.join(', ')}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{e.rows_processed}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{formatBytes(e.file_size)}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{new Date(e.uploaded_at).toLocaleString('vi-VN')}</td>
                      <td className="py-3 pr-4 dark:text-gray-300">{e.collection_name}</td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={async () => { if (confirm('Xóa dữ liệu Excel này?')) { await api.delete(`/admin/excel-files/${e.id}`); toast.success('Đã xóa'); await loadDocs() } }}
                            className="inline-flex items-center gap-1 text-red-600 hover:text-red-700 px-2 py-1 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20"
                          >
                            <Trash2 className="h-4 w-4" /> Xóa
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Confirm Delete Modal */}
      {isConfirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/30" onClick={() => !deleting && setConfirmOpen(false)} />
          <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
            <button className="absolute right-3 top-3 text-gray-400 hover:text-gray-600" onClick={() => !deleting && setConfirmOpen(false)}>
              <X className="h-4 w-4" />
            </button>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-red-100 flex items-center justify-center text-red-600">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Xóa tài liệu?</h3>
                <p className="text-sm text-gray-600">Bạn có chắc muốn xóa {docsToDelete.length} tài liệu đã chọn? Hành động này không thể hoàn tác.</p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-3">
              <button onClick={() => setConfirmOpen(false)} disabled={deleting} className="px-4 py-2 rounded border bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50">Hủy</button>
              <button onClick={handleConfirmDelete} disabled={deleting} className="px-4 py-2 rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50">
                {deleting ? 'Đang xóa...' : 'Xóa'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default AdminDocuments