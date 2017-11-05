/****************************************************************************
** Meta object code from reading C++ file 'MemWatchModel.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../GUI/MemWatcher/MemWatchModel.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MemWatchModel.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_MemWatchModel_t {
    QByteArrayData data[8];
    char stringdata0[101];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MemWatchModel_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MemWatchModel_t qt_meta_stringdata_MemWatchModel = {
    {
QT_MOC_LITERAL(0, 0, 13), // "MemWatchModel"
QT_MOC_LITERAL(1, 14, 11), // "writeFailed"
QT_MOC_LITERAL(2, 26, 0), // ""
QT_MOC_LITERAL(3, 27, 5), // "index"
QT_MOC_LITERAL(4, 33, 30), // "Common::MemOperationReturnCode"
QT_MOC_LITERAL(5, 64, 11), // "writeReturn"
QT_MOC_LITERAL(6, 76, 10), // "readFailed"
QT_MOC_LITERAL(7, 87, 13) // "dropSucceeded"

    },
    "MemWatchModel\0writeFailed\0\0index\0"
    "Common::MemOperationReturnCode\0"
    "writeReturn\0readFailed\0dropSucceeded"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MemWatchModel[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   29,    2, 0x06 /* Public */,
       6,    0,   34,    2, 0x06 /* Public */,
       7,    0,   35,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QModelIndex, 0x80000000 | 4,    3,    5,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void MemWatchModel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MemWatchModel *_t = static_cast<MemWatchModel *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->writeFailed((*reinterpret_cast< const QModelIndex(*)>(_a[1])),(*reinterpret_cast< Common::MemOperationReturnCode(*)>(_a[2]))); break;
        case 1: _t->readFailed(); break;
        case 2: _t->dropSucceeded(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (MemWatchModel::*_t)(const QModelIndex & , Common::MemOperationReturnCode );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MemWatchModel::writeFailed)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (MemWatchModel::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MemWatchModel::readFailed)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (MemWatchModel::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&MemWatchModel::dropSucceeded)) {
                *result = 2;
                return;
            }
        }
    }
}

const QMetaObject MemWatchModel::staticMetaObject = {
    { &QAbstractItemModel::staticMetaObject, qt_meta_stringdata_MemWatchModel.data,
      qt_meta_data_MemWatchModel,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *MemWatchModel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MemWatchModel::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_MemWatchModel.stringdata0))
        return static_cast<void*>(const_cast< MemWatchModel*>(this));
    return QAbstractItemModel::qt_metacast(_clname);
}

int MemWatchModel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QAbstractItemModel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void MemWatchModel::writeFailed(const QModelIndex & _t1, Common::MemOperationReturnCode _t2)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MemWatchModel::readFailed()
{
    QMetaObject::activate(this, &staticMetaObject, 1, Q_NULLPTR);
}

// SIGNAL 2
void MemWatchModel::dropSucceeded()
{
    QMetaObject::activate(this, &staticMetaObject, 2, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
